import os
import re
import matplotlib.pyplot as plt
import glob

# Constants
LOG_ROOT = r'd:\DinoV2-Unet-1\log'
DATASETS = ['clinicdb', 'etis', 'kvasir', 'colondb']
OUTPUT_FILE_LOSS = 'training_loss.png'
OUTPUT_FILE_DICE = 'validation_dice.png'
OUTPUT_FILE_IOU = 'validation_iou.png'

def get_latest_log_file(dataset_name):
    dataset_path = os.path.join(LOG_ROOT, dataset_name)
    if not os.path.exists(dataset_path):
        print(f"Directory not found: {dataset_path}")
        return None
    
    # Get all .log files
    log_files = glob.glob(os.path.join(dataset_path, '*.log'))
    if not log_files:
        print(f"No log files found in {dataset_path}")
        return None
    
    # Sort by modification time (or filename if timestamp is in filename)
    # Filenames are like '2025-12-30_12-01-05.log', so sorting by name works for finding latest
    log_files.sort(reverse=True)
    return log_files[0]

def parse_log_file(filepath):
    epochs = []
    train_losses = []
    val_dices = []
    val_ious = []
    
    # Regex pattern
    # Epoch 047/80 | time 60.2s | train: loss 0.0246 dice 0.9561 iou 0.9172 | val: loss 0.0388 mdice 0.9423 miou 0.8955
    pattern = re.compile(r'Epoch (\d+)/\d+.*?train: loss ([\d\.]+).*?val:.*?mdice ([\d\.]+) miou ([\d\.]+)')
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                val_dices.append(float(match.group(3)))
                val_ious.append(float(match.group(4)))
                
    return epochs, train_losses, val_dices, val_ious

def plot_metrics(all_data):
    # Set style
    plt.style.use('ggplot')
    
    metrics = [
        ('Training Loss', 'train_loss', OUTPUT_FILE_LOSS),
        ('Validation Dice', 'val_dice', OUTPUT_FILE_DICE),
        ('Validation IoU', 'val_iou', OUTPUT_FILE_IOU)
    ]
    
    for title, key, filename in metrics:
        plt.figure(figsize=(10, 6))
        
        for dataset in DATASETS:
            if dataset in all_data:
                data = all_data[dataset]
                # Determine which list to use based on key
                if key == 'train_loss':
                    y_values = data['train_losses']
                elif key == 'val_dice':
                    y_values = data['val_dices']
                elif key == 'val_iou':
                    y_values = data['val_ious']
                
                plt.plot(data['epochs'], y_values, label=dataset, linewidth=2)
        
        plt.title(f'{title} vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()

def main():
    all_data = {}
    
    for dataset in DATASETS:
        print(f"Processing {dataset}...")
        log_file = get_latest_log_file(dataset)
        if log_file:
            print(f"  Reading: {log_file}")
            epochs, train_losses, val_dices, val_ious = parse_log_file(log_file)
            if epochs:
                all_data[dataset] = {
                    'epochs': epochs,
                    'train_losses': train_losses,
                    'val_dices': val_dices,
                    'val_ious': val_ious
                }
                print(f"  Extracted {len(epochs)} epochs.")
            else:
                print("  No data found in file.")
        else:
            print(f"  Skipping.")
            
    if all_data:
        plot_metrics(all_data)
    else:
        print("No data extracted from any logs.")

if __name__ == "__main__":
    main()
