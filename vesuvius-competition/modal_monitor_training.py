"""
Monitor training progress without interrupting
"""
import modal

app = modal.App("vesuvius-monitor")
volume = modal.Volume.from_name("vesuvius-data")
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, volumes={"/mnt": volume})
def check_progress():
    """Check training progress from logs"""
    from pathlib import Path
    
    log_path = Path("/mnt/logs/training.log")
    if not log_path.exists():
        return "No training log found"
    
    # Read last 20 lines
    with open(log_path, 'r') as f:
        lines = f.readlines()
        last_lines = lines[-20:]
    
    # Find epoch info
    current_epoch = None
    last_val_loss = None
    
    for line in reversed(last_lines):
        if "Epoch" in line and "complete" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "Epoch":
                    current_epoch = parts[i+1]
                if part == "Val" and i+2 < len(parts):
                    last_val_loss = parts[i+2]
            break
    
    return {
        "last_lines": "".join(last_lines[-10:]),
        "current_epoch": current_epoch,
        "last_val_loss": last_val_loss
    }


@app.local_entrypoint()
def main():
    result = check_progress.remote()
    if isinstance(result, dict):
        print("\n" + "="*60)
        print("Training Progress Monitor")
        print("="*60)
        print(f"\nCurrent Epoch: {result['current_epoch'] or 'In progress'}")
        print(f"Last Val Loss: {result['last_val_loss'] or 'Not available'}")
        print("\nLast 10 log lines:")
        print(result['last_lines'])
    else:
        print(result)


if __name__ == "__main__":
    app.run()