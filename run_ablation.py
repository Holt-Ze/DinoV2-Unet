import os
import subprocess
import sys
from datetime import datetime

# Configuration
DATASETS = ['kvasir', 'clinicdb', 'colondb', 'etis']
# DATASETS = ['kvasir'] # Debug
LOG_ROOT = os.path.join(os.getcwd(), 'log', '消融实验')
PYTHON_EXE = sys.executable
SCRIPT_PATH = os.path.join(os.getcwd(), 'train.py')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_command(command, log_file, description):
    """Runs a command, prints output to console, and appends to log file."""
    print(f"--- Starting: {description} ---")
    print(f"Command: {command}")
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n\n{'='*80}\n")
        f.write(f"Experiment: {description}\n")
        f.write(f"Time: {datetime.now()}\n")
        f.write(f"Command: {command}\n")
        f.write(f"{'='*80}\n\n")

    # Use Popen to capture output in real-time
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        encoding='utf-8',
        errors='replace'
    )

    with open(log_file, 'a', encoding='utf-8') as f:
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                # Print to console
                sys.stdout.write(line)
                sys.stdout.flush()
                # Write to file
                f.write(line)
                f.flush()

    ret_code = process.poll()
    if ret_code != 0:
        print(f"!!! Error in {description}. Exit code: {ret_code}")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n!!! ERROR: Process exited with code {ret_code}\n")
    else:
        print(f"--- Finished: {description} ---\n")

def main():
    ensure_dir(LOG_ROOT)
    
    base_cmd = f'"{PYTHON_EXE}" "{SCRIPT_PATH}"'
    # Common flags to speed up ablation if needed (e.g., fewer epochs?) 
    # The user didn't specify, so we run with defaults or whatever is in train.py.
    # Assuming standard training.
    
    for dataset in DATASETS:
        print(f"\n\n################################################################")
        print(f"PROCESSING DATASET: {dataset.upper()}")
        print(f"################################################################")
        
        # --- Study 1: Optimizer Strategies ---
        log_file_s1 = os.path.join(LOG_ROOT, f"{dataset}_Study1_Strategies.log")
        # Clear log file if it exists? Or append? User implied "saving to...", usually new run overwrites or new file. 
        # But if we re-run, maybe append. Let's assume append or user manages it.
        # Actually, if I run the script twice, 'a' appends. That's safer.
        
        # 1.1 Frozen Encoder
        cmd = f"{base_cmd} --data {dataset} --optimizer-strategy frozen_encoder"
        run_command(cmd, log_file_s1, f"{dataset} - Study 1 - Frozen Encoder")
        
        # 1.2 Full Finetune
        cmd = f"{base_cmd} --data {dataset} --optimizer-strategy full_finetune"
        run_command(cmd, log_file_s1, f"{dataset} - Study 1 - Full Finetune")
        
        # 1.3 Partial Finetune (Baseline)
        cmd = f"{base_cmd} --data {dataset} --optimizer-strategy partial_finetune"
        run_command(cmd, log_file_s1, f"{dataset} - Study 1 - Partial Finetune")

        # --- Study 2: Pretrained Type ---
        log_file_s2 = os.path.join(LOG_ROOT, f"{dataset}_Study2_Pretraining.log")
        
        # 2.1 ImageNet Supervised
        cmd = f"{base_cmd} --data {dataset} --pretrained-type imagenet_supervised"
        run_command(cmd, log_file_s2, f"{dataset} - Study 2 - ImageNet Supervised")
        
        # 2.2 DINOv2 (Baseline - can we look at Study 1? User might want separate log.)
        # Running again for completeness and clean log.
        cmd = f"{base_cmd} --data {dataset} --pretrained-type dinov2"
        run_command(cmd, log_file_s2, f"{dataset} - Study 2 - DINOv2 (Self-Supervised)")

        # --- Study 3: Decoder Type ---
        log_file_s3 = os.path.join(LOG_ROOT, f"{dataset}_Study3_Decoder.log")
        
        # 3.1 Complex Decoder
        cmd = f"{base_cmd} --data {dataset} --decoder-type complex --optimizer-strategy partial_finetune"
        run_command(cmd, log_file_s3, f"{dataset} - Study 3 - Complex Decoder")
        
        # 3.2 Simple Decoder (Baseline)
        cmd = f"{base_cmd} --data {dataset} --decoder-type simple --optimizer-strategy partial_finetune"
        run_command(cmd, log_file_s3, f"{dataset} - Study 3 - Simple Decoder")

if __name__ == "__main__":
    main()
