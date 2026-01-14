#!/usr/bin/env python3
"""
Local preprocessing script - converts raw TIFFs to optimized .npz files
Run this on your laptop with 32GB RAM
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageSequence
from tqdm import tqdm
import argparse


def process_volume(volume_id, raw_dir, processed_dir, depth=65):
    """
    Process a single volume from competition data
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    
    # Paths to input files
    vol_file = raw_dir / "train_images" / f"{volume_id}.tif"
    mask_file = raw_dir / "train_labels" / f"{volume_id}.tif"
    
    if not vol_file.exists():
        print(f"⚠️  Volume file not found: {vol_file}")
        return False
    
    if not mask_file.exists():
        print(f"⚠️  Mask file not found: {mask_file}")
        return False
    
    print(f"📦 Processing Volume {volume_id}...")
    
    try:
        # Load mask (single image)
        mask = np.array(Image.open(mask_file)).astype(np.uint8)
        print(f"   Mask shape: {mask.shape}")
        
        # Load 3D volume from multi-page TIFF
        print(f"   Loading {depth} slices from TIFF...")
        with Image.open(vol_file) as img:
            slices = []
            for i, page in enumerate(ImageSequence.Iterator(img)):
                if i >= depth:
                    break
                slices.append(np.array(page).astype(np.uint16))
                if (i + 1) % 10 == 0:
                    print(f"   Loaded {i+1}/{depth} slices", end='\r')
        
        volume = np.stack(slices, axis=0)
        print(f"\n   Volume shape: {volume.shape}")
        
        # Calculate sizes
        vol_size_mb = volume.nbytes / (1024 * 1024)
        mask_size_mb = mask.nbytes / (1024 * 1024)
        print(f"   Volume size: {vol_size_mb:.1f} MB")
        print(f"   Mask size: {mask_size_mb:.1f} MB")
        
        # Save as compressed .npz
        output_vol = processed_dir / f"volume_{volume_id}.npz"
        output_mask = processed_dir / f"mask_{volume_id}.npz"
        
        print(f"   Compressing and saving...")
        np.savez_compressed(output_vol, data=volume)
        np.savez_compressed(output_mask, data=mask)
        
        # Check compressed sizes
        compressed_vol = output_vol.stat().st_size / (1024 * 1024)
        compressed_mask = output_mask.stat().st_size / (1024 * 1024)
        
        print(f"   Compressed volume: {compressed_vol:.1f} MB")
        print(f"   Compressed mask: {compressed_mask:.1f} MB")
        print(f"   Compression ratio: {vol_size_mb/compressed_vol:.1f}x")
        print(f"✅ Volume {volume_id} processed successfully\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {volume_id}: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Preprocess Vesuvius competition data")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/raw",
        help="Directory containing raw competition data"
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed .npz files"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=65,
        help="Number of slices to extract from each volume"
    )
    parser.add_argument(
        "--volume_ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific volume IDs to process (optional)"
    )
    
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect volume IDs if not specified
    if args.volume_ids is None:
        train_images_dir = raw_dir / "train_images"
        
        if not train_images_dir.exists():
            print(f"❌ Directory not found: {train_images_dir}")
            print(f"   Make sure you've downloaded the competition data to {raw_dir}")
            return
        
        volume_ids = [f.stem for f in train_images_dir.glob("*.tif")]
        
        if not volume_ids:
            print(f"❌ No .tif files found in {train_images_dir}")
            return
        
        print(f"🔍 Auto-detected {len(volume_ids)} volumes: {volume_ids[:5]}{'...' if len(volume_ids) > 5 else ''}\n")
    else:
        volume_ids = args.volume_ids
    
    # Process all volumes
    print(f"{'='*60}")
    print(f"Starting Preprocessing")
    print(f"{'='*60}")
    print(f"Input directory: {raw_dir}")
    print(f"Output directory: {processed_dir}")
    print(f"Depth: {args.depth} slices")
    print(f"Volumes to process: {len(volume_ids)}")
    print(f"{'='*60}\n")
    
    successful = 0
    failed = []
    
    for vid in tqdm(volume_ids, desc="Processing volumes"):
        if process_volume(vid, raw_dir, processed_dir, args.depth):
            successful += 1
        else:
            failed.append(vid)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Preprocessing Complete!")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {successful}/{len(volume_ids)}")
    if failed:
        print(f"❌ Failed: {len(failed)}")
        print(f"   Failed IDs: {failed}")
    print(f"📁 Output directory: {processed_dir}")
    print(f"{'='*60}\n")
    
    # Calculate total size
    processed_files = list(processed_dir.glob("*.npz"))
    total_size = sum(f.stat().st_size for f in processed_files)
    total_size_gb = total_size / (1024**3)
    
    print(f"📊 Storage Summary:")
    print(f"   Total files: {len(processed_files)}")
    print(f"   Total size: {total_size_gb:.2f} GB")
    print(f"   Average per volume: {total_size_gb/len(volume_ids)*2:.2f} GB (volume + mask)")
    print(f"\n✨ Ready for training!")


if __name__ == "__main__":
    main()