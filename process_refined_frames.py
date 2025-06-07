#!/usr/bin/env python3
"""
Process refined frames to match standard dimensions
"""

from pathlib import Path
from PIL import Image
import sys

class SimpleImageProcessor:
    def __init__(self):
        self.output_resolution = (1920, 1080)
        self.frames_dir = Path("./frames")
        self.processed_dir = Path("./processed")
        
        # Ensure output directory exists
        self.processed_dir.mkdir(exist_ok=True)
    
    def process_image(self, input_path: Path, output_path: Path) -> bool:
        """Resize image to standard resolution"""
        try:
            with Image.open(input_path) as img:
                # Resize to target resolution
                img_resized = img.resize(self.output_resolution, Image.Resampling.LANCZOS)
                
                # Save with high quality
                img_resized.save(output_path, quality=95, optimize=True)
                
                print(f"✓ Processed: {input_path.name} -> {output_path.name}")
                return True
                
        except Exception as e:
            print(f"✗ Error processing {input_path}: {e}")
            return False
    
    def process_all_frames(self):
        """Process all images in frames directory"""
        # Find all image files
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.frames_dir.glob(f"*{ext}"))
            image_files.extend(self.frames_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print("No images found in frames directory!")
            return
        
        print(f"Found {len(image_files)} images to process...")
        print(f"Output resolution: {self.output_resolution[0]}x{self.output_resolution[1]}")
        print("-" * 50)
        
        success_count = 0
        for img_path in sorted(image_files):
            # Create output filename
            output_path = self.processed_dir / img_path.name
            
            if self.process_image(img_path, output_path):
                success_count += 1
        
        print("-" * 50)
        print(f"Processing complete: {success_count}/{len(image_files)} successful")

if __name__ == "__main__":
    processor = SimpleImageProcessor()
    processor.process_all_frames()
