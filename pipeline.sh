#!/usr/bin/env python3
"""
Saint Lucifer Music Video Generation Pipeline
Handles AI image generation, processing, and video assembly
"""

import json
import os
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import requests
from PIL import Image, ImageEnhance
import cv2
import numpy as np

class MusicVideoConfig:
    """Configuration for the music video generation"""
    
    def __init__(self):
        # Project settings
        self.project_name = "saint_lucifer"
        self.song_duration = 199  # 3:19 in seconds
        self.fps = 24
        self.output_resolution = (1920, 1080)
        
        # Directory structure
        self.base_dir = Path(f"./{self.project_name}")
        self.images_dir = self.base_dir / "images"
        self.processed_dir = self.base_dir / "processed"
        self.frames_dir = self.base_dir / "frames"
        self.output_dir = self.base_dir / "output"
        
        # API settings (to be configured)
        self.midjourney_api_key = os.getenv("MIDJOURNEY_API_KEY")
        self.stability_api_key = os.getenv("STABILITY_API_KEY")
        
        # Style constants
        self.base_style = "film noir chiaroscuro lighting, wet pavement reflections, industrial urban decay"
        self.character_refs = {}
        
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories"""
        for dir_path in [self.images_dir, self.processed_dir, self.frames_dir, self.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

class SceneManager:
    """Manages scene definitions and timing"""
    
    def __init__(self):
        self.scenes = self._define_scenes()
    
    def _define_scenes(self) -> List[Dict]:
        """Define all scenes with timestamps and frame counts"""
        return [
            {
                "name": "intro",
                "start_time": 0.0,
                "end_time": 8.0,
                "frame_count": 10,
                "style": "extreme close-up, detailed woodgrain texture, stark black background",
                "description": "2x4 transformation sequence - lumber to club to cross to beam",
                "keyframes": [0, 2, 4, 6, 8]
            },
            {
                "name": "verse1",
                "start_time": 8.0,
                "end_time": 40.0,
                "frame_count": 40,
                "style": "sepia filter overlay, depression-era tint, geometric impossible architecture",
                "description": "Street building itself, character introductions, anagram sequence",
                "keyframes": [8, 16, 24, 32, 40]
            },
            {
                "name": "tag1",
                "start_time": 40.0,
                "end_time": 48.0,
                "frame_count": 8,
                "style": "morphing transition, shadow transformation",
                "description": "Narrator reaches for 2x4, shadow changes, street shifts",
                "keyframes": [40, 44, 48]
            },
            {
                "name": "verse2",
                "start_time": 48.0,
                "end_time": 80.0,
                "frame_count": 35,
                "style": "split screen visualization, warm vs cold lighting",
                "description": "Confessional style, world split between good and righteous",
                "keyframes": [48, 56, 64, 72, 80]
            },
            {
                "name": "tag2",
                "start_time": 80.0,
                "end_time": 88.0,
                "frame_count": 6,
                "style": "healing split with visible scar",
                "description": "Split world heals but leaves permanent division",
                "keyframes": [80, 84, 88]
            },
            {
                "name": "chorus1",
                "start_time": 88.0,
                "end_time": 116.0,
                "frame_count": 45,
                "style": "performance integration, theater metaphor, explosive moments",
                "description": "Banishment ritual, crumbling theater, shadow growing enormous",
                "keyframes": [88, 96, 104, 112, 116]
            },
            {
                "name": "tag3",
                "start_time": 116.0,
                "end_time": 124.0,
                "frame_count": 6,
                "style": "fluid transformation, melting buildings",
                "description": "Street morphs to lakeside road",
                "keyframes": [116, 120, 124]
            },
            {
                "name": "verse3",
                "start_time": 124.0,
                "end_time": 156.0,
                "frame_count": 35,
                "style": "watercolor impressionism, color saturation loss",
                "description": "Lakeside setting, role reversal, paper storm alibis",
                "keyframes": [124, 132, 140, 148, 156]
            },
            {
                "name": "chorus2",
                "start_time": 156.0,
                "end_time": 183.0,
                "frame_count": 40,
                "style": "kaleidoscope effects, cosmic scale shifts",
                "description": "Multiple reflections, vast cosmic space, final convergence",
                "keyframes": [156, 165, 174, 183]
            },
            {
                "name": "chorus3",
                "start_time": 183.0,
                "end_time": 195.0,
                "frame_count": 30,
                "style": "full saturation, explosive finale",
                "description": "All reflections collapse, bridge formation and crumbling",
                "keyframes": [183, 189, 195]
            },
            {
                "name": "outro",
                "start_time": 195.0,
                "end_time": 199.0,
                "frame_count": 8,
                "style": "line art dissolution, minimal",
                "description": "Characters dissolve to notes, street becomes blank page",
                "keyframes": [195, 197, 199]
            }
        ]
    
    def get_scene_by_time(self, timestamp: float) -> Optional[Dict]:
        """Get scene data for a specific timestamp"""
        for scene in self.scenes:
            if scene["start_time"] <= timestamp <= scene["end_time"]:
                return scene
        return None
    
    def get_total_frames(self) -> int:
        """Get total number of frames needed"""
        return sum(scene["frame_count"] for scene in self.scenes)

class PromptGenerator:
    """Generates consistent prompts for different scenes and characters"""
    
    def __init__(self, config: MusicVideoConfig):
        self.config = config
        self.character_prompts = self._define_characters()
        self.style_evolution = self._define_style_evolution()
    
    def _define_characters(self) -> Dict[str, str]:
        """Define character prompt templates"""
        return {
            "narrator": "middle-aged ordinary man, worn coat, average looking, nondescript clothes, expressionistic shadow that acts independently",
            "saint_lucifer": "tall gaunt figure, long dark coat, constantly shifting between angelic and demonic features, eyes catch light unnaturally, never settling on one form",
            "band_members": "animated musicians in basement club space, transparent walls casting shadows on pavement"
        }
    
    def _define_style_evolution(self) -> Dict[str, str]:
        """Define how visual style evolves through the song"""
        return {
            "intro": "detailed photorealistic woodgrain, stark black background",
            "verse1": "sepia depression-era tint, geometric impossible architecture, dream logic",
            "verse2": "split screen warm vs cold lighting, transparent figures fading",
            "chorus1": "full saturation explosive moments, theater crumbling metaphor",
            "verse3": "watercolor impressionism, color saturation loss, industrial lakeside",
            "chorus2": "kaleidoscope broken mirror effects, cosmic scale with street elements",
            "chorus3": "maximum saturation, explosive finale, bridge formation",
            "outro": "line art dissolution, musical notes floating, blank page minimal"
        }
    
    def generate_prompt(self, scene_name: str, frame_description: str, character: str = None) -> str:
        """Generate a complete prompt for a specific frame"""
        base_style = self.config.base_style
        scene_style = self.style_evolution.get(scene_name, "")
        character_desc = ""
        
        if character and character in self.character_prompts:
            character_desc = self.character_prompts[character]
        
        prompt_parts = [
            frame_description,
            character_desc,
            scene_style,
            base_style,
            "high quality, cinematic composition, dramatic lighting"
        ]
        
        # Filter out empty parts and join
        complete_prompt = ", ".join([part for part in prompt_parts if part.strip()])
        
        return complete_prompt

class ImageGenerator:
    """Handles AI image generation via various APIs"""
    
    def __init__(self, config: MusicVideoConfig):
        self.config = config
        self.session = requests.Session()
    
    def generate_via_stability(self, prompt: str, filename: str) -> bool:
        """Generate image using Stability AI API"""
        if not self.config.stability_api_key:
            print("Stability API key not configured")
            return False
        
        url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
        
        headers = {
            "Authorization": f"Bearer {self.config.stability_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "text_prompts": [{"text": prompt}],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 30
        }
        
        try:
            response = self.session.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            # Save image
            import base64
            image_data = base64.b64decode(data["artifacts"][0]["base64"])
            output_path = self.config.images_dir / filename
            
            with open(output_path, "wb") as f:
                f.write(image_data)
            
            print(f"Generated: {filename}")
            return True
            
        except Exception as e:
            print(f"Error generating image {filename}: {e}")
            return False
    
    def generate_batch(self, prompts: List[Dict]) -> List[str]:
        """Generate a batch of images"""
        successful_images = []
        
        for i, prompt_data in enumerate(prompts):
            filename = f"{prompt_data['scene']}_{prompt_data['frame']:03d}.png"
            
            if self.generate_via_stability(prompt_data['prompt'], filename):
                successful_images.append(filename)
                # Rate limiting
                time.sleep(1)
            else:
                print(f"Failed to generate frame {i}")
        
        return successful_images

class ImageProcessor:
    """Handles image processing and enhancement"""
    
    def __init__(self, config: MusicVideoConfig):
        self.config = config
    
    def enhance_image(self, input_path: Path, output_path: Path, scene_style: str) -> bool:
        """Apply scene-specific enhancements to images"""
        try:
            with Image.open(input_path) as img:
                # Resize to target resolution
                img = img.resize(self.config.output_resolution, Image.Resampling.LANCZOS)
                
                # Apply style-specific enhancements
                if "sepia" in scene_style:
                    img = self._apply_sepia(img)
                elif "split" in scene_style:
                    img = self._apply_split_effect(img)
                elif "watercolor" in scene_style:
                    img = self._apply_watercolor_effect(img)
                
                img.save(output_path, quality=95)
                return True
                
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False
    
    def _apply_sepia(self, img: Image.Image) -> Image.Image:
        """Apply sepia tone effect"""
        # Convert to numpy array for processing
        img_array = np.array(img)
        
        # Sepia transformation matrix
        sepia_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        sepia_img = img_array.dot(sepia_filter.T)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(sepia_img)
    
    def _apply_split_effect(self, img: Image.Image) -> Image.Image:
        """Apply split screen effect"""
        # This is a placeholder - would implement actual split screen logic
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(1.3)
    
    def _apply_watercolor_effect(self, img: Image.Image) -> Image.Image:
        """Apply watercolor effect"""
        # This is a placeholder - would implement actual watercolor logic
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(0.8)
    
    def interpolate_frames(self, frame1_path: Path, frame2_path: Path, 
                          output_dir: Path, num_interpolated: int = 3) -> List[Path]:
        """Create interpolated frames between two keyframes"""
        # This would use a tool like RIFE for frame interpolation
        # For now, just return the original frames
        return [frame1_path, frame2_path]

class VideoAssembler:
    """Handles final video assembly"""
    
    def __init__(self, config: MusicVideoConfig):
        self.config = config
    
    def create_video(self, frames_dir: Path, audio_path: Path, output_path: Path) -> bool:
        """Assemble final video with audio"""
        try:
            # Use FFmpeg to create video
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(self.config.fps),
                "-pattern_type", "glob",
                "-i", str(frames_dir / "*.png"),
                "-i", str(audio_path),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-pix_fmt", "yuv420p",
                "-shortest",
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Video created successfully: {output_path}")
                return True
            else:
                print(f"FFmpeg error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error creating video: {e}")
            return False
    
    def create_preview(self, frames_dir: Path, output_path: Path, 
                      start_frame: int = 0, num_frames: int = 120) -> bool:
        """Create a preview video of specific frames"""
        try:
            cmd = [
                "ffmpeg", "-y",
                "-start_number", str(start_frame),
                "-framerate", str(self.config.fps),
                "-i", str(frames_dir / "frame_%04d.png"),
                "-frames:v", str(num_frames),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            print(f"Error creating preview: {e}")
            return False

class MusicVideoPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self):
        self.config = MusicVideoConfig()
        self.scene_manager = SceneManager()
        self.prompt_generator = PromptGenerator(self.config)
        self.image_generator = ImageGenerator(self.config)
        self.image_processor = ImageProcessor(self.config)
        self.video_assembler = VideoAssembler(self.config)
    
    def generate_shot_list(self) -> List[Dict]:
        """Generate complete shot list with prompts"""
        shot_list = []
        
        for scene in self.scene_manager.scenes:
            scene_name = scene["name"]
            keyframes = scene["keyframes"]
            
            for i, timestamp in enumerate(keyframes):
                # Create frame description based on scene and position
                frame_desc = f"Scene {scene_name}, timestamp {timestamp}s"
                
                # Generate specific prompt
                prompt = self.prompt_generator.generate_prompt(
                    scene_name, 
                    frame_desc, 
                    character="narrator" if i % 2 == 0 else "saint_lucifer"
                )
                
                shot_list.append({
                    "scene": scene_name,
                    "timestamp": timestamp,
                    "frame": i,
                    "prompt": prompt,
                    "style": scene["style"],
                    "description": scene["description"]
                })
        
        return shot_list
    
    def save_shot_list(self, shot_list: List[Dict], filename: str = "shot_list.json"):
        """Save shot list to JSON file"""
        output_path = self.config.base_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(shot_list, f, indent=2)
        
        print(f"Shot list saved to {output_path}")
    
    def load_shot_list(self, filename: str = "shot_list.json") -> List[Dict]:
        """Load shot list from JSON file"""
        input_path = self.config.base_dir / filename
        
        if input_path.exists():
            with open(input_path, 'r') as f:
                return json.load(f)
        else:
            print(f"Shot list file not found: {input_path}")
            return []
    
    def generate_keyframes(self, shot_list: List[Dict]) -> bool:
        """Generate all keyframe images"""
        print(f"Generating {len(shot_list)} keyframes...")
        
        # Process in batches to avoid rate limits
        batch_size = 10
        
        for i in range(0, len(shot_list), batch_size):
            batch = shot_list[i:i + batch_size]
            successful = self.image_generator.generate_batch(batch)
            
            print(f"Batch {i//batch_size + 1}: {len(successful)}/{len(batch)} successful")
            
            # Wait between batches
            if i + batch_size < len(shot_list):
                time.sleep(5)
        
        return True
    
    def process_images(self) -> bool:
        """Process all generated images"""
        print("Processing images...")
        
        for scene in self.scene_manager.scenes:
            scene_style = scene["style"]
            
            # Process images for this scene
            scene_images = list(self.config.images_dir.glob(f"{scene['name']}_*.png"))
            
            for img_path in scene_images:
                output_path = self.config.processed_dir / img_path.name
                self.image_processor.enhance_image(img_path, output_path, scene_style)
        
        return True
    
    def create_final_frames(self) -> bool:
        """Create final frame sequence with interpolation"""
        print("Creating final frame sequence...")
        
        frame_counter = 0
        
        for scene in self.scene_manager.scenes:
            scene_images = sorted(list(self.config.processed_dir.glob(f"{scene['name']}_*.png")))
            
            for i, img_path in enumerate(scene_images):
                # Copy keyframe
                frame_name = f"frame_{frame_counter:04d}.png"
                output_path = self.config.frames_dir / frame_name
                
                # Copy file
                import shutil
                shutil.copy2(img_path, output_path)
                frame_counter += 1
                
                # Add interpolated frames if not the last frame in scene
                if i < len(scene_images) - 1:
                    next_img = scene_images[i + 1]
                    interpolated = self.image_processor.interpolate_frames(
                        img_path, next_img, self.config.frames_dir, 3
                    )
                    frame_counter += len(interpolated) - 2  # Subtract keyframes already counted
        
        print(f"Created {frame_counter} total frames")
        return True
    
    def run_full_pipeline(self, audio_path: str):
        """Run the complete pipeline"""
        print("Starting Saint Lucifer music video pipeline...")
        
        # Step 1: Generate shot list
        shot_list = self.generate_shot_list()
        self.save_shot_list(shot_list)
        print(f"Generated {len(shot_list)} shots")
        
        # Step 2: Generate keyframes
        if not self.generate_keyframes(shot_list):
            print("Failed to generate keyframes")
            return False
        
        # Step 3: Process images
        if not self.process_images():
            print("Failed to process images")
            return False
        
        # Step 4: Create final frame sequence
        if not self.create_final_frames():
            print("Failed to create frame sequence")
            return False
        
        # Step 5: Assemble video
        output_path = self.config.output_dir / "saint_lucifer_music_video.mp4"
        if not self.video_assembler.create_video(
            self.config.frames_dir, 
            Path(audio_path), 
            output_path
        ):
            print("Failed to create final video")
            return False
        
        print(f"Pipeline completed successfully! Video saved to {output_path}")
        return True

# Usage example and CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Saint Lucifer music video")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--shot-list-only", action="store_true", 
                       help="Only generate shot list")
    parser.add_argument("--from-shot-list", help="Load existing shot list file")
    parser.add_argument("--preview", action="store_true", 
                       help="Create preview video")
    
    args = parser.parse_args()
    
    pipeline = MusicVideoPipeline()
    
    if args.shot_list_only:
        shot_list = pipeline.generate_shot_list()
        pipeline.save_shot_list(shot_list)
        print("Shot list generated and saved")
    elif args.from_shot_list:
        shot_list = pipeline.load_shot_list(args.from_shot_list)
        if shot_list:
            # Continue from loaded shot list
            pipeline.generate_keyframes(shot_list)
            pipeline.process_images()
            pipeline.create_final_frames()
            
            output_path = pipeline.config.output_dir / "saint_lucifer_music_video.mp4"
            pipeline.video_assembler.create_video(
                pipeline.config.frames_dir, 
                Path(args.audio), 
                output_path
            )
    else:
        # Run full pipeline
        pipeline.run_full_pipeline(args.audio)
