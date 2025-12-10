"""
Azure ML Job Submission for Deformable DETR Training on extracted MITOS patches.
"""
from azure.ai.ml import command, Input, Output
from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential
from pathlib import Path
import json

# Training Configuration
EPOCHS = 50
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

print("[INFO] Connecting to Azure ML...")
credential = InteractiveBrowserCredential(tenant_id="5dbf1add-202a-4b8d-815b-bf0fb024e033")
ml_client = MLClient.from_config(credential=credential, path=config_path)

print(f"[INFO] Workspace: {config['workspace_name']}")
print(f"[INFO] Resource Group: {config['resource_group']}")

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
print(f"\nMonitor: https://ml.azure.com/runs/{returned_job.name}?wsid=/subscriptions/{config['subscription_id']}/resourcegroups/{config['resource_group']}/workspaces/{config['workspace_name']}")
