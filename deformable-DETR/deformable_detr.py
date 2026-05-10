import torch
import torch.nn.functional as F
from torch import nn
import math
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .backbone import build_backbone
from .matcher import build_matcher
from .losses import SetCriterion
from .deformable_transformer import DeformableTransformer
import copy


class BiFPNLayer(nn.Module):
    """
    One BiFPN layer: Top-down then bottom-up (fine to coarse) fusion
    """

    def __init__(self, num_levels: int, d_model: int, eps: float = 1e-4):
        super().__init__()
        self.num_levels = num_levels
        self.eps = eps

        # (num_levels-1) merge points per pass, blending 2 feature maps
        self.w_td = nn.Parameter(torch.ones(num_levels - 1, 2))
        self.w_bu = nn.Parameter(torch.ones(num_levels - 1, 2))

        def _dw_sep(d: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(d, d, 3, padding=1, groups=d, bias=False),
                nn.Conv2d(d, d, 1, bias=False),
                nn.GroupNorm(32, d),
                nn.ReLU(inplace=True),
            )

        self.conv_td = nn.ModuleList([_dw_sep(d_model) for _ in range(num_levels - 1)])
        self.conv_bu = nn.ModuleList([_dw_sep(d_model) for _ in range(num_levels - 1)])

    def _fuse(self, w_row: torch.Tensor, tensors: list) -> torch.Tensor:
        w = F.relu(w_row)
        w = w / (w.sum() + self.eps)
        return sum(w[i] * t for i, t in enumerate(tensors))

    def forward(self, features: list) -> list:
        N = self.num_levels
        td = [None] * N

        # Top-down: coarsest level unchanged, then upsample + fuse toward finest
        td[N - 1] = features[N - 1]
        for i in range(N - 2, -1, -1):
            m = N - 2 - i  # maps iteration order to a 0-based weight index
            up = F.interpolate(td[i + 1], size=features[i].shape[-2:], mode='nearest')
            td[i] = self.conv_td[m](self._fuse(self.w_td[m], [features[i], up]))

        # Bottom-up: finest td output unchanged, then pool + fuse toward coarsest
        out = [None] * N
        out[0] = td[0]
        for i in range(1, N):
            m = i - 1
            # max_pool for sparse mitosis signals (to keep peak activations)
            down = F.max_pool2d(out[i - 1], kernel_size=2, stride=2)
            if down.shape[-2:] != td[i].shape[-2:]:
                down = F.interpolate(down, size=td[i].shape[-2:], mode='nearest')
            out[i] = self.conv_bu[m](self._fuse(self.w_bu[m], [td[i], down]))

        return out


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection with deformable attention"""
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=False, num_bifpn_layers=0):
        """
        Args:
            - backbone : module of the backbone to be used.
            - transformer : module of the transformer.
            - num_classes : number of classes.
            - num_queries : the maximal number of objects Deformable DETR can detect in an image.
            - num_feature_levels : number of multi-scale levels.
            - aux_loss : True if auxiliary decoding losses are to be used.
            - num_bifpn_layers : number of BiFPN fusion layers applied after input_proj (0 = disabled).
        """
        super().__init__()
        self.transformer = transformer
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, self.transformer.C)

        self.class_pred = nn.Linear(transformer.C, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_pred.bias.data = torch.ones(num_classes) * bias_value
        num_pred = self.transformer.decoder.num_layers
        self.bbox_pred = MLP(transformer.C, transformer.C, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.backbone = backbone
        self.aux_loss = aux_loss
        if num_feature_levels > 1:
            self.input_proj = self.get_projections()
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], transformer.C, kernel_size=1),
                    nn.GroupNorm(32, transformer.C),
                )])


        # Each layer runs one top-down + one bottom-up fusion pass over all feature levels
        # num_bifpn_layers=0 disables BiFPN entirely
        if num_bifpn_layers > 0:
            self.bifpn = nn.ModuleList([
                BiFPNLayer(num_feature_levels, transformer.C)
                for _ in range(num_bifpn_layers)
            ])
        else:
            self.bifpn = None

        self.transformer.decoder.bbox_pred = None

    def get_projections(self):
        input_projections = []
        for _ in range(len(self.backbone.strides)):
            in_channels = self.backbone.num_channels[_]
            input_projections.append(nn.Sequential(
                nn.Conv2d(in_channels, self.transformer.C, kernel_size=1),
                nn.GroupNorm(32, self.transformer.C),
            ))
        for _ in range(self.num_feature_levels - len(self.backbone.strides)):
            input_projections.append(nn.Sequential(
                nn.Conv2d(in_channels, self.transformer.C, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, self.transformer.C),
            ))
            in_channels = self.transformer.C
        return nn.ModuleList(input_projections)

    def forward(self, samples):
        """
        Args:
            - samples : (tensor, mask) batch of images.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, poses = self.backbone(samples) # set of image features + positional encoding

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
        # For multi scale features
        if self.num_feature_levels > len(srcs):
            for l in range(len(srcs), self.num_feature_levels):
                if l == len(srcs):
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)

        # Apply BiFPN cross-scale fusion after projection (before transformer).
        if self.bifpn is not None:
            for bifpn_layer in self.bifpn:
                srcs = bifpn_layer(srcs)

        hs, ref_point, _ = self.transformer(srcs, masks, self.query_embed.weight, poses)
        hs = hs.transpose(1, 2).contiguous()
        ref_point = ref_point.transpose(0, 1).contiguous()
        inversed_ref_point = - torch.log(1 / (ref_point + 1e-10) - 1 + 1e-10)
        outputs_coord = self.bbox_pred(hs)
        outputs_coord[..., 0] = outputs_coord[..., 0] + inversed_ref_point[..., 0]
        outputs_coord[..., 1] = outputs_coord[..., 1] + inversed_ref_point[..., 1]
        outputs_coord = torch.sigmoid(outputs_coord)
        outputs_class = self.class_pred(hs)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] =  [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        return out
      
      
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Args:
            - outputs: raw outputs of the model.
            - target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        # Handle case where num_queries * num_classes < 100
        num_available = out_logits.shape[1] * out_logits.shape[2]
        k = min(100, num_available)
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), k, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Simple multi-layer perceptron"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 2  # Mitosis (0) + Look-alike (1)
    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = DeformableTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        scales=args.num_feature_levels,
        k=args.dec_n_points,
        last_height=args.last_height,
        last_width=args.last_width,
        use_zoom_in_bounding=not getattr(args, 'no_zoom_in_bounding', False),
        use_zero_bias_init=not getattr(args, 'no_zero_bias_init', False),
        scale_factors_list=getattr(args, 'scale_factors', None)
    )

    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        num_bifpn_layers=getattr(args, 'num_bifpn_layers', 0),
    )
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    return model, criterion, postprocessors
