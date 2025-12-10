from azure.ai.ml import command, Input, Output
from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential
from pathlib import Path
import json

# Configuration
PATCHES_PER_SLIDE = 200  # Configurable - adjust for experiments
PATCH_SIZE = 256
TRAIN_VAL_SPLIT = 0.8

# Sampling ratios
MITOSIS_PROB = 0.45
LOOKALIKE_PROB = 0.45
BACKGROUND_PROB = 0.10

# Azure ML Configuration
COMPUTE_NAME = "xseligam-mitos-train"
ENVIRONMENT = "xseligam_mitos:17"
RAW_DATA_URI = "azureml:xseligam-ccmct-wsi:1"

# Load config for display
config_path = Path(__file__).parent / "config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

# Connect to Azure ML using from_config (reads config.json automatically)
print("[INFO] Prihlasujem sa do Azure...")
credential = InteractiveBrowserCredential(tenant_id="5dbf1add-202a-4b8d-815b-bf0fb024e033")
ml_client = MLClient.from_config(credential=credential, path=config_path)

print(f"[INFO] Connected to workspace: {config['workspace_name']}")
print(f"[INFO] Resource group: {config['resource_group']}")

# Get code directory (parent directory where extract_wsi_patches.py is located)
code_dir = str(Path(__file__).parent.resolve())

# Build command - simple direct python call
job = command(
    code=code_dir,
    
    command=(
        "python extract_wsi_patches.py "
        "--wsi_dir ${{inputs.wsi_data}}/WSI "
        "--annotation_json ${{inputs.wsi_data}}/databases/MITOS_WSI_CCMCT_HEAEL.sqlite "
        "--output_dir ${{outputs.patch_dataset}} "
        f"--patches_per_slide {PATCHES_PER_SLIDE} "
        f"--patch_size {PATCH_SIZE} "
        f"--train_val_split {TRAIN_VAL_SPLIT} "
        f"--mitosis_prob {MITOSIS_PROB} "
        f"--lookalike_prob {LOOKALIKE_PROB} "
        f"--background_prob {BACKGROUND_PROB} "
        "--seed 42"
    ),
    
    inputs={
        "wsi_data": Input(
            type="uri_folder",
            path=RAW_DATA_URI,
            mode="ro_mount"
        ),
    },
    
    outputs={
        "patch_dataset": Output(
            type="uri_folder",
            mode="upload"
        ),
    },
    
    environment=ENVIRONMENT,
    compute=COMPUTE_NAME,
    display_name=f"extract-patches-{PATCHES_PER_SLIDE}pps",
    experiment_name="Mitos-Patch-Extraction",
)

# Submit job
print("=" * 60)
print("WSI Patch Extraction Job")
print("=" * 60)
print(f"Patches per slide: {PATCHES_PER_SLIDE}")
print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
print(f"Sampling: {MITOSIS_PROB*100:.0f}% mitosis, "
      f"{LOOKALIKE_PROB*100:.0f}% look-alikes, "
      f"{BACKGROUND_PROB*100:.0f}% background")
print(f"Compute: {COMPUTE_NAME}")
print(f"Environment: {ENVIRONMENT}")
print("=" * 60)

print("\n[INFO] Submitting job to Azure ML...")
returned_job = ml_client.jobs.create_or_update(job)

print(f"\n[SUCCESS] Job submitted!")
print(f"Job Name: {returned_job.name}")
print(f"Status: {returned_job.status}")
print(f"\nMonitor at: https://ml.azure.com/runs/{returned_job.name}?wsid=/subscriptions/{config['subscription_id']}/resourcegroups/{config['resource_group']}/workspaces/{config['workspace_name']}")

