#!/usr/bin/env python3
"""
Download model from Modal and convert to a format that works without smp
Run this LOCALLY where you have segmentation_models_pytorch installed
"""
import modal
import torch
import segmentation_models_pytorch as smp
import numpy as np
from pathlib import Path

app = modal.App("download-and-convert-model")
volume = modal.Volume.from_name("vesuvius-data")

@app.function(
    volumes={"/mnt": volume},
    image=modal.Image.debian_slim()
        .pip_install(["torch", "numpy"])
)
def download_checkpoint():
    """Download the checkpoint from Modal"""
    checkpoint_path = Path("/mnt/models/balanced_best.pth")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint
    else:
        print("Checkpoint not found!")
        return None

@app.local_entrypoint()
def main():
    print("Downloading checkpoint from Modal...")
    checkpoint = download_checkpoint.remote()
    
    if checkpoint is None:
        print("Failed to download checkpoint")
        return
    
    # Save raw checkpoint
    torch.save(checkpoint, "modal_checkpoint_raw.pth")
    print("Saved raw checkpoint")
    
    # Load with smp locally
    print("\nCreating smp model locally...")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=32,
        classes=1,
        activation=None
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("Model loaded successfully!")
    
    # Test inference locally
    print("\nTesting inference...")
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 32, 128, 128)
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
    
    # Export to ONNX (works without smp)
    print("\nExporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        "vesuvius_model.onnx",
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}}
    )
    print("Exported to vesuvius_model.onnx")
    
    # Also save as TorchScript
    print("\nExporting to TorchScript...")
    scripted_model = torch.jit.script(model)
    scripted_model.save("vesuvius_model.pt")
    print("Exported to vesuvius_model.pt")
    
    print("\n✅ Done! Upload vesuvius_model.onnx or vesuvius_model.pt to Kaggle")

if __name__ == "__main__":
    app.run()