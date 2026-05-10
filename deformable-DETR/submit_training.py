from azure.ai.ml import command, Input, Output
from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential
from pathlib import Path
from datetime import datetime
import json

# Training Configuration
EPOCHS = 150
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
NUM_QUERIES = 5 # Number of experts that predict
ENC_LAYERS = 3
DEC_LAYERS = 3

# Azure ML Configuration
COMPUTE_NAME = "xseligam-mitos-train"
ENVIRONMENT = "xseligam_mitos:17"

PATCH_DATASET_URI = "azureml:azureml_nice_dinner_nkclz1yd1s_output_data_patch_dataset:1"

# Setup
config_path = Path(__file__).parent / "config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

print("Connecting to Azure ML...")
credential = InteractiveBrowserCredential(tenant_id="5dbf1add-202a-4b8d-815b-bf0fb024e033")
ml_client = MLClient.from_config(credential=credential, path=config_path)

print(f"Workspace: {config['workspace_name']}")
print(f"Resource Group: {config['resource_group']}")

# Get code directory
code_dir = str(Path(__file__).parent.resolve())

job = command(
    code=code_dir,
    
    command=(
        "python train_azure.py "
        "--data_path ${{inputs.patch_dataset}} "
        "--output_dir ${{outputs.model_output}} "
        f"--epochs {EPOCHS} "
        f"--batch_size {BATCH_SIZE} "
        f"--lr {LEARNING_RATE} "
        f"--num_queries {NUM_QUERIES} "
        f"--enc_layers {ENC_LAYERS} "
        f"--dec_layers {DEC_LAYERS} "
        "--dataset_file mitos"
    ),
    
    environment_variables={
        "WANDB_API_KEY": "b45c181c94b978711d21388c4d8f7c03ca6d06d2",
        "WANDB_PROJECT": "Mitos_BP_DeformableDETR",
    },
    
    inputs={
        "patch_dataset": Input(
            type="uri_folder",
            path=PATCH_DATASET_URI,
            mode="ro_mount"
        ),
    },
    
    outputs={
        "model_output": Output(
            type="uri_folder",
            mode="upload"
        ),
    },
    
    environment=ENVIRONMENT,
    compute=COMPUTE_NAME,
    display_name=f"deformable-detr-e{EPOCHS}-bs{BATCH_SIZE}",
    experiment_name="Mitos-DeformableDETR-Training",
)

print("\n" + "=" * 60)
print("Deformable DETR Training Job")
print("=" * 60)
print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Num Queries: {NUM_QUERIES}")
print(f"Encoder/Decoder Layers: {ENC_LAYERS}/{DEC_LAYERS}")
print(f"Dataset: {PATCH_DATASET_URI}")
print(f"Compute: {COMPUTE_NAME}")
print(f"Environment: {ENVIRONMENT}")
print("=" * 60)

print("\n[INFO] Submitting training job...")
returned_job = ml_client.jobs.create_or_update(job)

print(f"\n[SUCCESS] Job submitted!")
print(f"Job Name: {returned_job.name}")
print(f"Status: {returned_job.status}")
print(f"\nRun in Azure ML Studio:")
print(f"{returned_job.studio_url}")

# Save configuration locally for research tracking
experiments_dir = Path(__file__).parent / "experiments"
experiments_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
config_filename = f"{timestamp}_{returned_job.name}.txt"
config_filepath = experiments_dir / config_filename

with open(config_filepath, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("DEFORMABLE DETR - EXPERIMENT CONFIGURATION\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(f"Job Name: {returned_job.name}\n")
    f.write(f"Submitted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Status: {returned_job.status}\n\n")
    
    f.write("TRAINING\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Num Queries: {NUM_QUERIES}\n")
    f.write(f"Encoder Layers: {ENC_LAYERS}\n")
    f.write(f"Decoder Layers: {DEC_LAYERS}\n\n")
    
    f.write("ARCHITECTURE\n")
    f.write("Backbone: resnet101\n")
    f.write("Hidden Dim: 128\n")
    f.write("FFN Dim: 512\n")
    f.write("Attention Heads: 4\n")
    f.write("Warmup Epochs: 10\n\n")
    
    f.write("AUGMENTATIONS\n")
    f.write("- RandomHorizontalFlip (p=0.5)\n")
    f.write("- RandomVerticalFlip (p=0.5)\n")
    f.write("- RandomRotation90 (p=0.75)\n")
    f.write("- RandomColorJitter\n")
    f.write("- RandomStainAugmentation (LAB space)\n\n")
    
    f.write("EVALUATION\n")
    f.write("Score Threshold: 0.35\n")
    f.write("IoU Threshold: 0.5\n\n")
    
    f.write("AZURE\n")
    f.write(f"Compute: {COMPUTE_NAME}\n")
    f.write(f"Environment: {ENVIRONMENT}\n")
    f.write(f"Dataset: {PATCH_DATASET_URI}\n\n")

print(f"Configuration saved locally: {config_filepath}")
