#!/usr/bin/env python3
"""
Saint Lucifer Music Video Generation Pipeline
Handles AI image generation, processing, and video assembly
"""

from dotenv import load_dotenv
load_dotenv()

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
import shutil

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
        self.midjourney_api_key = os.getenv("MIDJOURNEY_API_KEY") # Note: Midjourney integration will require further code changes
        self.stability_api_key = os.getenv("STABILITY_API_KEY")
        self.replicate_api_token = os.getenv("REPLICATE_API_TOKEN") # Added Replicate token
        
        # Style constants
        self.base_style = "film noir chiaroscuro lighting, wet pavement reflections, industrial urban decay"
        self.character_refs = {} # This could be used for image references later
        
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
        """Define all scenes with timestamps and frame counts, including specific keyframe prompts"""
        return [
            {
                "name": "intro",
                "start_time": 0.0,
                "end_time": 8.0,
                "frame_count": 10, # Target frame count for this scene
                "style": "extreme close-up, detailed woodgrain texture, stark black background",
                "description": "2x4 transformation sequence - lumber to club to cross to beam",
                "keyframes": [
                    {"timestamp": 0, "prompt_suffix": "of raw lumber. A single, smooth, dark wooden 2x4."},
                    {"timestamp": 2, "prompt_suffix": "of a wooden club, clearly identifiable as morphing from lumber, with a wicked, bent nail gleaming at one end."},
                    {"timestamp": 4, "prompt_suffix": "of a wooden cross, intricately carved, morphing from a club."},
                    {"timestamp": 6, "prompt_suffix": "of a raw construction beam, morphing from a cross."},
                    {"timestamp": 8, "prompt_suffix": "of raw lumber with a bent nail gleaming wickedly, appearing to have transformed back from a construction beam."}
                ]
            },
            {
                "name": "verse1",
                "start_time": 8.0,
                "end_time": 40.0,
                "frame_count": 40,
                "style": "sepia filter overlay, depression-era tint, geometric impossible architecture",
                "description": "Street building itself, character introductions, anagram sequence",
                "keyframes": [
                    {"timestamp": 8, "prompt_suffix": "A dark street building itself around a fallen 2x4, buildings growing like plants, streetlights blooming like flowers, but with wrong geometry."},
                    {"timestamp": 16, "prompt_suffix": "An average middle-aged man (Narrator) materializing from line art to full color on the street, his disproportionately large and menacing shadow stretches behind him. Focus on Narrator."},
                    {"timestamp": 24, "prompt_suffix": "Saint Lucifer unfolding from the shadows between buildings like origami in reverse, his form constantly shifting between angelic and demonic, never quite settling. Focus on Saint Lucifer."},
                    {"timestamp": 32, "prompt_suffix": "Saint Lucifer extending his hand, a 2x4 floating towards the Narrator, candy-colored ribbons swirling menacingly around it, creating beautiful but menacing patterns."},
                    {"timestamp": 40, "prompt_suffix": "Letters pouring from Saint Lucifer's mouth like water, flowing across the ground, rearranging into words and scrambling like insects before they can be read. Saint Lucifer walks away in the background. A deep sharp silence follows him, visualized as expanding circles of absolute black silence that consume all color and sound, pulsing and vibrating with overwhelming visual intensity."}
                ]
            },
            {
                "name": "tag1",
                "start_time": 40.0,
                "end_time": 48.0,
                "frame_count": 8,
                "style": "morphing transition, shadow transformation",
                "description": "Narrator reaches for 2x4, shadow changes, street shifts",
                "keyframes": [
                    {"timestamp": 40, "prompt_suffix": "The Narrator reaching for the 2x4, his fingers just about to touch it, his shadow starting to grow more angular and aggressive."},
                    {"timestamp": 44, "prompt_suffix": "The Narrator's shadow visibly changing shape, growing larger and more angular and aggressive. The street itself shifting from Depression-era to a more modern industrial look."},
                    {"timestamp": 48, "prompt_suffix": "The street fully shifted to modern industrial. The Narrator's shadow is enormous and aggressive, clearly holding a weapon. The Narrator looks conflicted."}
                ]
            },
            {
                "name": "verse2",
                "start_time": 48.0,
                "end_time": 80.0,
                "frame_count": 35,
                "style": "split screen visualization, warm vs cold lighting",
                "description": "Confessional style, world split between good and righteous",
                "keyframes": [
                    {"timestamp": 48, "prompt_suffix": "Narrator addressing floating text bubbles that represent 'you', they bob and weave around him like living things. His shadow, now clearly holding a weapon, contradicts his spoken words. Confessional style."},
                    {"timestamp": 56, "prompt_suffix": "All other figures on the street becoming transparent, fading in and out of existence. Only the 'you' text bubbles remain solid and vibrant around the Narrator."},
                    {"timestamp": 64, "prompt_suffix": "The screen literally splits down the middle. Left side shows the same street bathed in warm, soft light with rounded, organic shapes. Right side shows harsh geometric angles and cold blue-white light. The Narrator stands precisely on the dividing line, one foot in each world, being pulled apart."},
                    {"timestamp": 72, "prompt_suffix": "The Narrator's struggle on the dividing line between the good and righteous plan, being pulled between two visually distinct worlds."},
                    {"timestamp": 80, "prompt_suffix": "Close up on the Narrator, still straddling the divide, showing the internal conflict on his face, the split world visible behind him."}
                ]
            },
            {
                "name": "tag2",
                "start_time": 80.0,
                "end_time": 88.0,
                "frame_count": 6,
                "style": "healing split with visible scar",
                "description": "Split world heals but leaves permanent division",
                "keyframes": [
                    {"timestamp": 80, "prompt_suffix": "The split screen healing and merging back into a single view of the street, but leaving a visible, faint scar down the middle of the environment."},
                    {"timestamp": 84, "prompt_suffix": "The street now appears whole, but with a noticeable linear scar running through all structures and ground."},
                    {"timestamp": 88, "prompt_suffix": "Overhead shot of the scarred street, indicating the lasting division."}
                ]
            },
            {
                "name": "chorus1",
                "start_time": 88.0,
                "end_time": 116.0,
                "frame_count": 45,
                "style": "performance integration, theater metaphor, explosive moments",
                "description": "Banishment ritual, crumbling theater, shadow growing enormous",
                "keyframes": [
                    {"timestamp": 88, "prompt_suffix": "Band members appearing as animated characters playing in a space that's simultaneously a basement club AND the street. Walls are transparent, instruments cast shadows on the pavement. Narrator looking determined."},
                    {"timestamp": 96, "prompt_suffix": "The Narrator waving the 2x4 like a conductor's baton. With each gesture, pieces of Saint Lucifer begin to break away like puzzle pieces, floating toward the edges of the frame."},
                    {"timestamp": 104, "prompt_suffix": "The street transforming into a crumbling theater. Saint Lucifer is on a stage that's visibly breaking apart. Red curtains try to close but they're tattered and won't meet in the middle."},
                    {"timestamp": 112, "prompt_suffix": "The Narrator's shadow has grown enormous, taking up most of the screen. It swings the shadow-2x4 with immense force, shattering the theater illusion."},
                    {"timestamp": 116, "prompt_suffix": "Only the empty, scarred street remains after the illusion shatters. The Narrator's enormous shadow dominates the frame."}
                ]
            },
            {
                "name": "tag3",
                "start_time": 116.0,
                "end_time": 124.0,
                "frame_count": 6,
                "style": "fluid transformation, melting buildings",
                "description": "Street morphs to lakeside road",
                "keyframes": [
                    {"timestamp": 116, "prompt_suffix": "The street fluid-like morphing into a lakeside road. Buildings melting and reforming as shoreline features."},
                    {"timestamp": 120, "prompt_suffix": "Mid-transformation, showing the fluidity of the urban landscape giving way to a natural shoreline, with hints of industrial elements still present."},
                    {"timestamp": 124, "prompt_suffix": "A fully transformed lakeside road, shoreline features clearly visible. The reflection on the lake is not the sky, but the industrial cityscape from earlier verses."}
                ]
            },
            {
                "name": "verse3",
                "start_time": 124.0,
                "end_time": 156.0,
                "frame_count": 35,
                "style": "watercolor impressionism, color saturation loss, industrial lakeside",
                "description": "Lakeside setting, role reversal, paper storm alibis",
                "keyframes": [
                    {"timestamp": 124, "prompt_suffix": "Now in a completely different animation style, more impressionistic, like moving watercolors. The lakeside road reflecting the industrial cityscape. Saint Lucifer sits on a bench, losing color saturation, becoming more like a pencil sketch."},
                    {"timestamp": 132, "prompt_suffix": "The Narrator approaches Saint Lucifer, who is now very desaturated and ghost-like. Narrator is fully colored and solid, his shadow massive and dark, indicating role reversal."},
                    {"timestamp": 140, "prompt_suffix": "The Narrator's mouth opens and sheets of paper fly out like a blizzard. Each paper shows a different excuse/alibi, but they're all contradictory. The papers swirl around and dissolve."},
                    {"timestamp": 148, "prompt_suffix": "A deep loud silence follows the Narrator, visualized as heavy, dark blue rain that falls upward from the ground to the sky. Each raindrop contains a tiny reflection of the Narrator's regretful face, watercolor impressionism."},
                    {"timestamp": 156, "prompt_suffix": "Narrator walks away, his massive shadow casting a long, dark path, leaving the desaturated Saint Lucifer behind on the bench."}
                ]
            },
            {
                "name": "chorus2",
                "start_time": 156.0,
                "end_time": 183.0,
                "frame_count": 40,
                "style": "kaleidoscope broken mirror effects, cosmic scale with street elements",
                "description": "Multiple reflections, vast cosmic space, final convergence",
                "keyframes": [
                    {"timestamp": 156, "prompt_suffix": "Multiple versions of both the Narrator and Saint Lucifer appearing, reflected and refracted across the screen like a broken mirror. Each reflection shows a different past moment from their encounters."},
                    {"timestamp": 165, "prompt_suffix": "The characters become tiny figures in a vast cosmic space. Stars and planets move around them, but the celestial bodies are made of distinct street elements like lampposts, manhole covers, and brick walls."},
                    {"timestamp": 174, "prompt_suffix": "The cosmic space is filled with swirling street elements, the tiny figures of Narrator and Saint Lucifer moving towards each other for a final convergence."},
                    {"timestamp": 183, "prompt_suffix": "All the scattered reflections collapse back into two distinct figures: the Narrator and Saint Lucifer, facing each other directly."}
                ]
            },
            {
                "name": "chorus3",
                "start_time": 183.0,
                "end_time": 195.0,
                "frame_count": 30,
                "style": "full saturation, explosive finale, bridge formation",
                "description": "All reflections collapse, bridge formation and crumbling",
                "keyframes": [
                    {"timestamp": 183, "prompt_suffix": "The Narrator and Saint Lucifer face each other. The 2x4 transforms one final time, into a glowing bridge connecting them, momentarily. Full saturation, explosive finale."},
                    {"timestamp": 189, "prompt_suffix": "The 2x4 bridge crumbling into sawdust between the Narrator and Saint Lucifer. The sawdust drifts and dissipates."},
                    {"timestamp": 195, "prompt_suffix": "Narrator and Saint Lucifer stand looking at each other, the remnants of the sawdust bridge between them, indicating the end of their overt conflict. Full saturation, explosive finale."}
                ]
            },
            {
                "name": "outro",
                "start_time": 195.0,
                "end_time": 199.0,
                "frame_count": 8,
                "style": "line art dissolution, musical notes floating, blank page minimal",
                "description": "Characters dissolve to notes, street becomes blank page",
                "keyframes": [
                    {"timestamp": 195, "prompt_suffix": "Both characters dissolving into line art, then further into musical notes that float away into a vast, empty space. The street beneath them begins to become a blank, white page. Minimalist style."},
                    {"timestamp": 197, "prompt_suffix": "The street has become a blank page. Only a single piece of ordinary lumber remains on the page, no bent nail, no menace."},
                    {"timestamp": 199, "prompt_suffix": "Extreme close-up on the single piece of ordinary lumber. The wood grain of the 2x4 subtly forms into the song title 'Saint Lucifer', then gently fades to black, leaving a blank screen."}
                ]
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
            "band_members": "animated musicians in a basement club space, transparent walls casting shadows on pavement"
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
    
    def generate_prompt(self, scene_name: str, keyframe_prompt_suffix: str, character: Optional[str] = None) -> str:
        """Generate a complete prompt for a specific frame"""
        base_style = self.config.base_style
        scene_style = self.style_evolution.get(scene_name, "")
        character_desc = ""
        
        # Only include character description if character is NOT None and is defined
        if character:
            character_parts = []
            for c in character.split(','):
                c_stripped = c.strip()
                if c_stripped in self.character_prompts:
                    character_parts.append(self.character_prompts[c_stripped])
            character_desc = ", ".join([p for p in character_parts if p])
        
        prompt_parts = [
            keyframe_prompt_suffix, # Use the specific prompt suffix here
            character_desc, # This will be empty if character is None or not found
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
            "height": 1024, # SDXL default
            "width": 1024,  # SDXL default
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
            
    def generate_via_replicate(self, prompt: str, filename: str) -> bool:
        """Generate image using Replicate API (placeholder)"""
        if not self.config.replicate_api_token:
            print("Replicate API token not configured")
            return False
        
        # This is a simplified placeholder.
        # You'd need to specify a model, version, and input for the actual Replicate API call.
        # Example (replace with actual model you intend to use):
        # url = "https://api.replicate.com/v1/predictions"
        # headers = {
        #     "Authorization": f"Token {self.config.replicate_api_token}",
        #     "Content-Type": "application/json"
        # }
        # payload = {
        #     "version": "YOUR_MODEL_VERSION_ID_HERE", # e.g., "stability-ai/stable-diffusion:ac732df8f9efaace7965a60a8d05a41151acef0b60d700be8ab485174a72cdb3"
        #     "input": {"prompt": prompt}
        # }
        # response = self.session.post(url, headers=headers, json=payload)
        # Handle response to get image URL and download it.
        
        print(f"Skipping Replicate generation for {filename} (placeholder).")
        return False


    def generate_batch(self, prompts_data: List[Dict]) -> List[str]:
        """Generate a batch of images using primary API or fallback"""
        successful_images = []
        
        for i, prompt_data in enumerate(prompts_data):
            # Use global_frame_index for unique, sequential filenames
            filename = f"frame_{prompt_data['global_frame_index']:04d}_{prompt_data['scene']}.png"
            
            # Prioritize Stability AI
            if self.generate_via_stability(prompt_data['prompt'], filename):
                successful_images.append(filename)
            # Add fallback to Replicate if Stability AI fails or if you prefer a different primary
            # elif self.generate_via_replicate(prompt_data['prompt'], filename):
            #    successful_images.append(filename)
            else:
                print(f"Failed to generate frame {filename} via primary API.")
            
            # Rate limiting
            time.sleep(1) # Adjust based on API rate limits and cost goals
                
        return successful_images

class ImageProcessor:
    """Handles image processing and enhancement"""
    
    def __init__(self, config: MusicVideoConfig):
        self.config = config
    
    def enhance_image(self, input_path: Path, output_path: Path, scene_style: str) -> bool:
        """Apply scene-specific enhancements and resize images"""
        try:
            with Image.open(input_path) as img:
                # Resize to target resolution, maintaining aspect ratio or cropping
                # For now, a simple resize. You might want to implement letterboxing or cropping
                img = img.resize(self.config.output_resolution, Image.Resampling.LANCZOS)
                
                # Apply style-specific enhancements
                if "sepia" in scene_style:
                    img = self._apply_sepia(img)
                elif "split" in scene_style:
                    img = self._apply_split_effect(img) # Placeholder for actual split effect
                elif "watercolor" in scene_style:
                    img = self._apply_watercolor_effect(img) # Placeholder for actual watercolor effect
                # Add more style effects as needed (e.g., color saturation, noir filters)
                
                img.save(output_path, quality=95)
                return True
                
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False
    
    def _apply_sepia(self, img: Image.Image) -> Image.Image:
        """Apply sepia tone effect"""
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
        """Placeholder for applying a split screen effect or contrast enhancement"""
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(1.3)
    
    def _apply_watercolor_effect(self, img: Image.Image) -> Image.Image:
        """Placeholder for applying a watercolor effect"""
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(0.8)
    
    def interpolate_frames(self, frame1_path: Path, frame2_path: Path,
                           output_dir: Path, num_interpolated: int = 3) -> List[Path]:
        """
        Create interpolated frames between two keyframes using RIFE.
        This method will be executed on the cloud instance.
        """
        print(f"Interpolating between {frame1_path.name} and {frame2_path.name} with {num_interpolated} frames in between.")

        temp_input_dir = output_dir / f"rife_input_{frame1_path.stem}_{frame2_path.stem}"
        temp_output_dir = output_dir / f"rife_output_{frame1_path.stem}_{frame2_path.stem}"
        temp_input_dir.mkdir(exist_ok=True)
        temp_output_dir.mkdir(exist_ok=True)

        # Copy the two keyframes to the temporary input directory
        # Naming them 0.png and 1.png for RIFE's standard input
        shutil.copy(frame1_path, temp_input_dir / "0.png")
        shutil.copy(frame2_path, temp_input_dir / "1.png")

        interpolated_paths = []
        
        # Calculate RIFE's 'exp' parameter: 2^exp - 1 == num_interpolated
        rife_exp = 0
        if num_interpolated > 0:
            # Find the smallest 'exp' that generates at least num_interpolated frames
            rife_exp = int(np.ceil(np.log2(num_interpolated + 1)))

        if rife_exp > 0:
            # Path to RIFE inference script relative to where pipeline.py runs
            # Assuming ECCV2022-RIFE is in the same directory as pipeline.py (/workspace/)
            rife_script_path = "ECCV2022-RIFE/inference_img.py"
            # Set the path to your downloaded RIFE model (the directory containing rife_v4.25.pth)
            rife_model_path = "ECCV2022-RIFE/train_log/rife_v4.25" # <-- THIS IS THE CORRECT PATH FOR YOUR MODEL

            cmd = [
                "python", rife_script_path,
                "--exp", str(rife_exp),
                "--img", str(temp_input_dir),
                "--output", str(temp_output_dir),
                "--model_path", rife_model_path # Crucial for RIFE to find its weights
            ]

            try:
                print(f"Running RIFE command: {' '.join(cmd)}")
                # Use a higher timeout for subprocess.run as RIFE can take time
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600) # 10 minute timeout
                print(f"RIFE stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"RIFE stderr:\n{result.stderr}")

                rife_output_frames = sorted(list(temp_output_dir.glob("*.png")))
                
                # RIFE typically outputs 2^exp frames, including copies of the original inputs.
                # We want the 'num_interpolated' frames in between.
                # Assuming the first and last frames in RIFE's output are the original inputs,
                # we extract the intermediate frames and then trim to num_interpolated if needed.
                if len(rife_output_frames) >= (2**rife_exp): # Ensure RIFE generated enough total frames
                    intermediate_frames = rife_output_frames[1:-1] # Skips the first (0.png) and last (1.png) which are original copies
                    interpolated_paths = intermediate_frames[:num_interpolated] # Trim to exactly what we need
                else:
                    print(f"Warning: RIFE did not produce enough frames. Expected at least {2**rife_exp} total, got {len(rife_output_frames)}. No interpolated frames copied for this segment.")

            except subprocess.CalledProcessError as e:
                print(f"RIFE execution failed with error code {e.returncode}. Stderr:\n{e.stderr}")
                print(f"Command executed: {' '.join(cmd)}")
            except FileNotFoundError:
                print(f"Error: RIFE script not found. Check path: {rife_script_path}. Or model path: {rife_model_path}")
            except subprocess.TimeoutExpired:
                print(f"Error: RIFE command timed out after 600 seconds for {frame1_path.name} to {frame2_path.name}.")
            except Exception as e:
                print(f"Error during RIFE interpolation: {e}")

        # Clean up temporary directories
        shutil.rmtree(temp_input_dir, ignore_errors=True)
        shutil.rmtree(temp_output_dir, ignore_errors=True)

        return interpolated_paths

class VideoAssembler:
    """Handles final video assembly"""
    
    def __init__(self, config: MusicVideoConfig):
        self.config = config
    
    def create_video(self, frames_dir: Path, audio_path: Path, output_path: Path) -> bool:
        """Assemble final video with audio"""
        try:
            # Use FFmpeg to create video
            # Assumes frames are named sequentially, e.g., frame_0000.png, frame_0001.png
            cmd = [
                "ffmpeg", "-y", # -y to overwrite output files without asking
                "-framerate", str(self.config.fps),
                "-i", str(frames_dir / "frame_%04d.png"), # Input image sequence
                "-i", str(audio_path), # Input audio
                "-c:v", "libx264",
                "-c:a", "aac",
                "-pix_fmt", "yuv420p", # Pixel format for wide compatibility
                "-shortest", # Finish encoding when the shortest input stream ends (audio or video)
                str(output_path)
            ]
            
            print(f"Running FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True) # check=True raises error for non-zero exit codes
            
            print(f"Video created successfully: {output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error (code {e.returncode}): {e.stderr}")
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
            
            print(f"Running FFmpeg preview command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.returncode == 0
            
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg preview error (code {e.returncode}): {e.stderr}")
            return False
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
        global_frame_index = 0 # Track global frame index for unique filenames
        
        for scene in self.scene_manager.scenes:
            scene_name = scene["name"]
            
            for i, keyframe_data in enumerate(scene["keyframes"]): # Iterate through keyframe_data dictionaries
                timestamp = keyframe_data["timestamp"]
                prompt_suffix = keyframe_data["prompt_suffix"] # Get the specific suffix
                
                # Determine character(s) based on the scene and specific keyframe context
                character = None # Initialize character as None for every frame
                
                if scene_name == "intro":
                    character = None # Explicitly no characters in intro
                elif scene_name == "verse1":
                    if "Narrator" in prompt_suffix and "Saint Lucifer" not in prompt_suffix:
                        character = "narrator"
                    elif "Saint Lucifer" in prompt_suffix and "Narrator" not in prompt_suffix:
                        character = "saint_lucifer"
                    elif "Narrator" in prompt_suffix and "Saint Lucifer" in prompt_suffix:
                        character = "narrator, saint_lucifer"
                    # If neither are explicitly mentioned, character remains None (e.g., for street building)
                elif scene_name == "tag1":
                    character = "narrator" # Focus on Narrator and his shadow
                elif scene_name == "verse2":
                    character = "narrator" # Focus on Narrator, "you" bubbles are environmental
                elif scene_name == "chorus1":
                    if "Band members" in prompt_suffix:
                        character = "narrator, band_members, saint_lucifer" # Band appears with characters
                    elif "Narrator's shadow" in prompt_suffix or "Narrator waving" in prompt_suffix:
                        character = "narrator"
                    elif "Saint Lucifer is on a stage" in prompt_suffix:
                        character = "saint_lucifer"
                    else:
                        character = "narrator, saint_lucifer" # Default if both are implied
                elif scene_name == "verse3":
                    character = "narrator, saint_lucifer" # Both typically present, role reversal
                elif scene_name in ["chorus2", "chorus3"]:
                        character = "narrator, saint_lucifer" # Both interacting/converging/dissolving
                elif scene_name == "outro":
                    if "Both characters dissolving" in prompt_suffix:
                        character = "narrator, saint_lucifer"
                    else:
                        character = None # No characters in final wood/title frames

                # Generate specific prompt
                prompt = self.prompt_generator.generate_prompt(
                    scene_name,
                    prompt_suffix,
                    character=character
                )
                
                shot_list.append({
                    "scene": scene_name,
                    "timestamp": timestamp,
                    "global_frame_index": global_frame_index,
                    "frame_index_in_scene": i,
                    "prompt": prompt,
                    "style": scene["style"],
                    "description": prompt_suffix
                })
                global_frame_index += 1
        
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
        
        # Ensure image directories are clean before generation if starting fresh
        # (This is commented out, uncomment if you want to clear previous images)
        # for f in self.config.images_dir.iterdir():
        #    if f.is_file(): f.unlink()

        # Process in batches to avoid rate limits
        batch_size = 5 # Small batch size for cautious testing
        
        for i in range(0, len(shot_list), batch_size):
            batch = shot_list[i:i + batch_size]
            print(f"\n--- Generating batch {i//batch_size + 1}/{len(shot_list)//batch_size + (1 if len(shot_list)%batch_size > 0 else 0)} ---")
            successful = self.image_generator.generate_batch(batch)
            
            print(f"Batch {i//batch_size + 1}: {len(successful)}/{len(batch)} successful")
            
            # Wait between batches
            if i + batch_size < len(shot_list):
                print(f"Waiting for API rate limit...")
                time.sleep(5) # Adjust based on API rate limits and cost goals
        
        return True
    
    def process_images(self) -> bool:
        """Process all generated images"""
        print("Processing images...")
        
        # Ensure processed_dir is clean
        # (This is commented out, uncomment if you want to clear previous processed images)
        # for f in self.config.processed_dir.iterdir():
        #    if f.is_file(): f.unlink()

        # Get all generated images from the images_dir and sort them by global_frame_index
        all_generated_images = []
        # Load the shot list to map filenames back to original prompt data
        shot_list_loaded = self.load_shot_list()
        if not shot_list_loaded:
            print("Cannot process images: shot_list.json not found or empty.")
            return False

        # Create a mapping from generated filename stem to scene data
        filename_to_scene_map = {}
        for item in shot_list_loaded:
            filename_stem = f"frame_{item['global_frame_index']:04d}_{item['scene']}"
            filename_to_scene_map[filename_stem] = {"scene_name": item['scene'], "style": item['style'], "global_frame_index": item['global_frame_index']}

        for img_path in self.config.images_dir.glob("*.png"):
            # Extract the stem to match against our map
            stem_parts = img_path.stem.split('_')
            if len(stem_parts) >= 3 and stem_parts[0] == 'frame' and stem_parts[1].isdigit():
                current_stem = "_".join(stem_parts[0:3]) # e.g., "frame_0000_intro"
                if current_stem in filename_to_scene_map:
                    img_data = filename_to_scene_map[current_stem]
                    all_generated_images.append({
                        "path": img_path,
                        "scene_name": img_data["scene_name"],
                        "style": img_data["style"],
                        "global_frame_index": img_data["global_frame_index"]
                    })
        
        # Sort by global_frame_index to ensure correct processing order
        all_generated_images.sort(key=lambda x: x["global_frame_index"])

        for img_data in all_generated_images:
            input_path = img_data["path"]
            output_filename = f"processed_frame_{img_data['global_frame_index']:04d}.png" # Standardized processed name
            output_path = self.config.processed_dir / output_filename
            self.image_processor.enhance_image(input_path, output_path, img_data["style"])
        
        print(f"Processed {len(all_generated_images)} images.")
        return True
    
    def create_final_frames(self) -> bool:
        """Create final frame sequence with interpolation and consistent naming"""
        print("Creating final frame sequence...")
        
        # Ensure frames_dir is clean
        # (This is commented out, uncomment if you want to clear previous final frames)
        # for f in self.config.frames_dir.iterdir():
        #    if f.is_file(): f.unlink()

        processed_images = sorted(list(self.config.processed_dir.glob("processed_frame_*.png")))
        
        if not processed_images:
            print("No processed images found to create final frames.")
            return False

        final_frame_count = 0
        
        # Load the full shot list to determine desired interpolation counts per segment
        # This is critical as interpolation counts can vary by scene/transition
        # Assuming shot_list.json is in the base_dir (e.g., saint_lucifer/)
        full_shot_list = self.load_shot_list(filename="shot_list.json")
        
        # Map global_frame_index to shot data for easy lookup
        shot_data_map = {shot['global_frame_index']: shot for shot in full_shot_list}

        # Copy the first processed image as the very first final frame
        first_keyframe_path = processed_images[0]
        shutil.copy2(first_keyframe_path, self.config.frames_dir / f"frame_{final_frame_count:04d}.png")
        final_frame_count += 1
        print(f"Copied keyframe {first_keyframe_path.name} to frame_{final_frame_count-1:04d}.png")


        for i in range(len(processed_images) - 1): # Iterate up to second to last image
            current_keyframe_path = processed_images[i]
            next_keyframe_path = processed_images[i+1]
            
            # Extract global_frame_index from filenames to get shot data
            current_gfi = int(current_keyframe_path.stem.split('_')[2])
            next_gfi = int(next_keyframe_path.stem.split('_')[2])

            # Determine the number of interpolated frames for this segment based on timestamps and FPS
            current_timestamp = shot_data_map.get(current_gfi, {}).get('timestamp')
            next_timestamp = shot_data_map.get(next_gfi, {}).get('timestamp')

            if current_timestamp is not None and next_timestamp is not None:
                duration_between_keyframes = next_timestamp - current_timestamp
                # Calculate total frames *needed* in this segment (including the next keyframe)
                # If duration is 0, this will be 0, num_interpolated_between_keyframes will be -1 (max 0) -> 0
                total_frames_in_segment = round(duration_between_keyframes * self.config.fps)
                
                # The number of frames *between* the current and next keyframe
                num_interpolated_between_keyframes = max(0, total_frames_in_segment - 1)
            else:
                print(f"Warning: Missing timestamp for keyframe {current_gfi} or {next_gfi}. Using default interpolation (3 frames).")
                num_interpolated_between_keyframes = 3 # Default if data is missing

            # Call RIFE through the ImageProcessor
            interpolated_files = self.image_processor.interpolate_frames(
                current_keyframe_path, next_keyframe_path, self.config.frames_dir, num_interpolated_between_keyframes
            )
            
            # Copy/rename interpolated frames to the final sequential naming scheme
            for interp_idx, interp_file_path in enumerate(interpolated_files):
                # We need to copy from the temporary path where RIFE outputs, to the final frames_dir
                shutil.copy2(interp_file_path, self.config.frames_dir / f"frame_{final_frame_count:04d}.png")
                final_frame_count += 1
            
            # Copy the next keyframe as a final frame *after* any interpolations.
            shutil.copy2(next_keyframe_path, self.config.frames_dir / f"frame_{final_frame_count:04d}.png")
            final_frame_count += 1
            print(f"Copied keyframe {next_keyframe_path.name} to frame_{final_frame_count-1:04d}.png")
            
        print(f"Created {final_frame_count} total frames for video assembly.")
        return True
    
    def run_full_pipeline(self, audio_path: str):
        """Run the complete pipeline"""
        print("Starting Saint Lucifer music video pipeline...")
        
        # Step 1: Generate shot list
        shot_list = self.generate_shot_list()
        self.save_shot_list(shot_list)
        print(f"Generated {len(shot_list)} shots")
        
        # Step 2: Generate keyframes
        # We assume keyframes are already generated locally and uploaded to /processed
        # So we skip direct generation from API on the pod unless explicitly asked.
        # If you need to generate keyframes on the pod, uncomment the line below.
        # if not self.generate_keyframes(shot_list):
        #     print("Failed to generate keyframes")
        #     return False
        
        # Step 3: Process images (resize, apply filters)
        # We assume processed images are already uploaded
        # If you need to process raw images (from images_dir) on the pod, uncomment the line below.
        # if not self.process_images():
        #     print("Failed to process images")
        #     return False
        
        print("Skipping keyframe generation and image processing (assuming processed images are pre-uploaded).")

        # Step 4: Create final frame sequence (including interpolation)
        if not self.create_final_frames():
            print("Failed to create frame sequence (interpolation failed)")
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
    parser.add_argument("--audio", required=True, help="Path to audio file (e.g., saint_lucifer.mp3)")
    parser.add_argument("--shot-list-only", action="store_true",
                        help="Only generate shot list (no image generation or video assembly)")
    parser.add_argument("--from-shot-list", help="Load existing shot list file instead of generating a new one")
    parser.add_argument("--preview", action="store_true",
                        help="Create a short preview video from existing processed frames (first 120 frames)")
    parser.add_argument("--generate-only-keyframes", action="store_true",
                        help="Only generate keyframes and process them, do not create final frames or video.")
    
    args = parser.parse_args()
    
    pipeline = MusicVideoPipeline()
    
    # --- CLI Logic ---
    if args.shot_list_only:
        shot_list = pipeline.generate_shot_list()
        pipeline.save_shot_list(shot_list)
        print("Shot list generated and saved.")
    elif args.preview:
        # For preview, we assume frames have already been created
        print("Attempting to create preview video from generated frames...")
        preview_output_path = pipeline.config.output_dir / "saint_lucifer_preview.mp4"
        # Ensure frames are available in pipeline.config.frames_dir
        if not pipeline.config.frames_dir.exists() or not any(pipeline.config.frames_dir.iterdir()):
            print("Error: No frames found in 'frames' directory for preview. Please run full pipeline first.")
        else:
            pipeline.video_assembler.create_preview(
                pipeline.config.frames_dir,
                preview_output_path,
                num_frames=24 * 10 # 10 seconds of preview
            )
            print(f"Preview video saved to {preview_output_path}")
    elif args.generate_only_keyframes:
        print("Generating keyframes and processing them (no final video assembly)...")
        shot_list = pipeline.load_shot_list(args.from_shot_list) if args.from_shot_list else pipeline.generate_shot_list()
        if shot_list:
            pipeline.generate_keyframes(shot_list)
            pipeline.process_images()
            print("Keyframe generation and processing complete.")
        else:
            print("Shot list is empty. Cannot generate keyframes.")
    else:
        # Full pipeline run
        # If --from-shot-list is used, we load it, otherwise generate a new one.
        # This will assume keyframes and processed images are pre-uploaded for the full run.
        # If you need to generate/process on the pod, adjust run_full_pipeline's logic.
        pipeline.run_full_pipeline(args.audio)
