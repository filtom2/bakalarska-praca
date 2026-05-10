import time
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
NUM_QUERIES = 10 # Number of experts that predict (def. 5)
ENC_LAYERS = 3
DEC_LAYERS = 3

# Azure ML Configuration
COMPUTE_NAME = "xseligam-mitos-train"
ENVIRONMENT = "xseligam_mitos:17"

PATCH_DATASET_URI = "azureml:azureml_nice_dinner_nkclz1yd1s_output_data_patch_dataset:1"

# Ablation Experiments
# E0 is assumed to be the default run already completed.
ABLATION_EXPERIMENTS = {
    "E1_no_zoom_in": {"--no_zoom_in_bounding": True, "--no_zero_bias_init": True},
    "E2_no_zero_bias": {"--no_zero_bias_init": True},
    "E3_no_bounding": {"--no_zoom_in_bounding": True},
    "E4_uniform_scale": {"--scale_factors": "8.0 8.0 8.0 8.0"},
    "E5_no_stain_aug": {"--no_stain_aug": True},
}

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
experiments_dir = Path(__file__).parent / "experiments"
experiments_dir.mkdir(exist_ok=True)

print(f"\nSubmitting {len(ABLATION_EXPERIMENTS)} ablation experiments to Azure ML...")

for exp_id, exp_flags in ABLATION_EXPERIMENTS.items():
    print(f"\n{'-' * 60}")
    print(f"Preparing experiment: {exp_id}")
    print(f"{'-' * 60}")
    
    # Build command string
    base_cmd = (
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
    )
    
    extra_cmd = []
    for k, v in exp_flags.items():
        if isinstance(v, bool) and v:
            extra_cmd.append(k)
        else:
            extra_cmd.append(f"{k} {v}")
            
    full_cmd = base_cmd + " " + " ".join(extra_cmd)
    
    escaped_env = ENVIRONMENT.replace(":", "_")
    job = command(
        code=code_dir,
        command=full_cmd,
        environment_variables={
            "WANDB_API_KEY": "b45c181c94b978711d21388c4d8f7c03ca6d06d2",
            "WANDB_PROJECT": "Mitos_BP_DeformableDETR",
            "WANDB_RUN_NAME": f"ablation-{exp_id.replace('_', '-')}",
        },
        inputs={
            "patch_dataset": Input(type="uri_folder", path=PATCH_DATASET_URI, mode="ro_mount"),
        },
        outputs={
            "model_output": Output(type="uri_folder", mode="upload"),
        },
        environment=ENVIRONMENT,
        compute=COMPUTE_NAME,
        display_name=f"ablation-{exp_id}-e{EPOCHS}",
        experiment_name="Mitos-DeformableDETR-Ablation",
    )
    
    print(f"Command: {full_cmd}")
    
    # Submit job
    try:
        returned_job = ml_client.jobs.create_or_update(job)
        print(f"[SUCCESS] Job {exp_id} submitted!")
        print(f"Job Name: {returned_job.name}")
        print(f"{returned_job.studio_url}")
        
        # Save configuration locally
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_filename = f"{timestamp}_{exp_id}_{returned_job.name}.txt"
        config_filepath = experiments_dir / config_filename
        
        with open(config_filepath, 'w') as f:
            f.write(f"EXPERIMENT: {exp_id}\n")
            f.write(f"Job Name: {returned_job.name}\n")
            f.write(f"Status: {returned_job.status}\n")
            f.write(f"Command: {full_cmd}\n")
            
    except Exception as e:
        print(f"[ERROR] Failed to submit {exp_id}: {e}")
        
    print("Waiting a bit before next submission...")
    time.sleep(5)
    
print("\nAll submissions complete.")
