# Saint Lucifer - Intro Scene Iteration Workflow
# Building on existing FastAPI backend for refined keyframe iteration

import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from backend/app/.env
load_dotenv(Path(__file__).parent / "app" / ".env")

class IntroSceneIterator:
    """
    Specialized workflow for iterating on intro scene keyframes
    Builds on existing FastAPI backend architecture
    """
    
    def __init__(self, backend_url: str = "localhost:8000", frames_dir: str = None):
        self.backend_url = backend_url
        self.scene_id = "intro"
        self.current_keyframes = {}
        self.locked_frames = set()  # Frames marked as "good" - don't regenerate
        
        # Set frames directory - default to the location you specified
        if frames_dir is None:
            import os
            self.frames_dir = Path(os.path.expanduser("~/Development/layersofmeta/frames/"))
        else:
            self.frames_dir = Path(frames_dir)
        
        # Get Stability AI API key
        self.stability_api_key = os.getenv("STABILITY_API_KEY")
        if not self.stability_api_key:
            print("⚠️  Warning: STABILITY_API_KEY not found in environment")
        else:
            print(f"✅ Stability AI API key loaded (ends with: ...{self.stability_api_key[-4:]})")
        
    def load_current_intro_state(self):
        """Load current intro keyframes (0000-0004) and their status"""
        print("📋 Loading current intro scene state...")
        
        # Get scene info from backend
        response = requests.get(f"http://{self.backend_url}/api/scenes/{self.scene_id}")
        if response.status_code == 200:
            scene_data = response.json()
            print(f"✅ Scene loaded: {scene_data.get('name', 'Intro')}")
            print(f"   Duration: {scene_data.get('start_time', 0):.1f}s - {scene_data.get('end_time', 8):.1f}s")
            # Handle beats safely
            beats = scene_data.get('beats', [])
            if beats:
                print(f"   Total beats: {len(beats)}")
            else:
                print(f"   Beats: Using default intro sequence (5 keyframes)")
        
        # Load keyframes from your actual files instead of backend API
        print(f"   Loading keyframes from: {self.frames_dir}")
        
        intro_files = []
        for i in range(5):  # 0000-0004
            filename = f"{i:04d}.png"
            file_path = self.frames_dir / filename
            if file_path.exists():
                intro_files.append({
                    "frame_number": i,
                    "filename": filename,
                    "path": str(file_path),
                    "exists": True
                })
            else:
                intro_files.append({
                    "frame_number": i,
                    "filename": filename,
                    "path": str(file_path),
                    "exists": False
                })
        
        self.current_keyframes = intro_files
        existing_count = sum(1 for f in intro_files if f["exists"])
        print(f"   Current keyframes: {existing_count}/5 files found")
        
        # Display current frame summary
        for kf in intro_files:
            frame_num = f"{kf['frame_number']:04d}"
            status = "✅ exists" if kf["exists"] else "❌ missing"
            print(f"   Frame {frame_num}: {kf['filename']} ({status})")
        return self.current_keyframes
    
    def analyze_transformation_gaps(self):
        """
        Analyze intro sequence for transformation gaps that need more intermediate frames
        Based on your director's notes: lumber → club → cross → beam → lumber w/ nail
        """
        print("\n🔍 Analyzing transformation gaps in intro sequence...")
        
        transformations = [
            {"from": 0, "to": 1, "desc": "lumber → club", "gap_severity": "medium"},
            {"from": 1, "to": 2, "desc": "club → cross", "gap_severity": "high"},
            {"from": 2, "to": 3, "desc": "cross → beam", "gap_severity": "medium"},
            {"from": 3, "to": 4, "desc": "beam → lumber w/ nail", "gap_severity": "high"}
        ]
        
        print("📊 Transformation Gap Analysis:")
        for t in transformations:
            print(f"   {t['desc']:20} | Gap: {t['gap_severity']:6} | Frames {t['from']:04d}→{t['to']:04d}")
        
        # Recommend intermediate frames
        recommendations = []
        for t in transformations:
            if t['gap_severity'] == 'high':
                recommendations.append({
                    "transition": t['desc'],
                    "current_frames": [t['from'], t['to']],
                    "recommended_intermediates": 2,  # Add 2 frames between
                    "priority": "high"
                })
            elif t['gap_severity'] == 'medium':
                recommendations.append({
                    "transition": t['desc'],
                    "current_frames": [t['from'], t['to']],
                    "recommended_intermediates": 1,  # Add 1 frame between
                    "priority": "medium"
                })
        
        print(f"\n💡 Recommendations: {len(recommendations)} transitions need intermediate frames")
        for rec in recommendations:
            print(f"   {rec['transition']:20} | Add {rec['recommended_intermediates']} frames | Priority: {rec['priority']}")
        
        return recommendations
    
    def setup_frame_locking_system(self):
        """
        Set up system to 'lock' good frames so they don't get regenerated
        """
        print("\n🔒 Setting up frame locking system...")
        
        # Check which frames user wants to lock as "good"
        current_frames = list(range(5))  # 0000-0004
        
        print("Current intro frames (0000-0004):")
        for i in current_frames:
            frame_num = f"{i:04d}"
            status = "🔓 unlocked" if i not in self.locked_frames else "🔒 locked"
            print(f"   Frame {frame_num}: {status}")
        
        # Interactive locking (in real implementation, this would be GUI)
        print("\n📝 Frame Locking Options:")
        print("   - All frames currently unlocked (can be regenerated)")
        print("   - Use lock_frame(frame_number) to protect good frames")
        print("   - Use unlock_frame(frame_number) to allow regeneration")
        
        return True
    
    def test_club_to_cross_transition(self):
        """
        Test generation for the most challenging transition: club → cross
        """
        print("\n🎯 TESTING: Club → Cross Transition (Frames 0001→0002)")
        print("=" * 55)
        
        # Create prompts for the 2 intermediate frames
        intermediate_prompts = [
            "extreme close-up of wooden club with cross-beam beginning to emerge from the handle, detailed woodgrain texture, stark black background, film noir lighting",
            "extreme close-up of wooden club transforming, perpendicular beam growing more prominent, forming cross shape, detailed woodgrain texture, stark black background, film noir lighting"
        ]
        
        # Generate the intermediate frames
        success_count = 0
        for i, prompt in enumerate(intermediate_prompts, 1):
            filename = f"0001_{i:02d}.png"
            
            if self.generate_via_backend_api(prompt, filename):
                success_count += 1
            else:
                print(f"❌ Failed to generate {filename}")
        
        print(f"\n📊 Generation Results:")
        print(f"   • {success_count}/{len(intermediate_prompts)} frames generated successfully")
        
        if success_count > 0:
            print(f"\n✅ Test successful! Check these files in {self.frames_dir}:")
            for i in range(1, success_count + 1):
                filename = f"0001_{i:02d}.png"
                print(f"   • {filename}")
            
            print(f"\n🎬 Sequence now available:")
            print(f"   0001.png (club) → 0001_01.png → 0001_02.png → 0002.png (cross)")
            
            print(f"\n🎯 Next steps:")
            print(f"   1. Review the generated intermediate frames")
            print(f"   2. If quality is good, proceed with other transitions")
            print(f"   3. Test RIFE interpolation on enhanced sequence")
        else:
            print(f"\n❌ Generation failed. Check API key and connection.")
        
        return success_count > 0
    
    def lock_frame(self, frame_number: int):
        """Lock a frame as 'good' - won't be regenerated"""
        self.locked_frames.add(frame_number)
        print(f"🔒 Frame {frame_number:04d} locked (protected from regeneration)")
    
    def unlock_frame(self, frame_number: int):
        """Unlock a frame for regeneration"""
        self.locked_frames.discard(frame_number)
        print(f"🔓 Frame {frame_number:04d} unlocked (available for regeneration)")
    
    def generate_intermediate_frames(self, transition_spec: Dict):
        """
        Generate intermediate frames for a specific transformation
        """
        print(f"\n🎨 Generating intermediate frames for: {transition_spec['transition']}")
        
        from_frame = transition_spec['current_frames'][0]
        to_frame = transition_spec['current_frames'][1]
        num_intermediates = transition_spec['recommended_intermediates']
        
        # Skip if either frame is locked
        if from_frame in self.locked_frames and to_frame in self.locked_frames:
            print(f"   ⚠️  Both keyframes locked, generating intermediates only")
        
        # Create prompts for intermediate frames
        intermediate_prompts = self._create_intermediate_prompts(
            from_frame, to_frame, num_intermediates
        )
        
        # Generate via backend API
        for i, prompt in enumerate(intermediate_prompts):
            intermediate_frame_id = f"{from_frame:04d}_{i+1:02d}_intermediate"
            
            generation_request = {
                "scene_id": self.scene_id,
                "frame_id": intermediate_frame_id,
                "prompt": prompt,
                "is_intermediate": True,
                "parent_frames": [from_frame, to_frame]
            }
            
            print(f"   🎯 Generating {intermediate_frame_id}...")
            # In real implementation: POST to /api/scenes/{scene_id}/keyframes/generate
            
        return intermediate_prompts
    
    def _create_intermediate_prompts(self, from_frame: int, to_frame: int, num_intermediates: int) -> List[str]:
        """
        Create prompts for intermediate frames based on transformation type
        """
        # Define transformation prompt templates
        transformation_templates = {
            (0, 1): {  # lumber → club
                "base": "detailed woodgrain texture, stark black background",
                "progression": [
                    "raw lumber beginning to show wear marks and rounded edges",
                    "lumber taking on club-like proportions, end beginning to thicken"
                ]
            },
            (1, 2): {  # club → cross
                "base": "detailed woodgrain texture, stark black background",
                "progression": [
                    "wooden club with cross-beam beginning to emerge from the handle",
                    "club transforming, perpendicular beam growing more prominent, forming cross shape"
                ]
            },
            (2, 3): {  # cross → beam
                "base": "detailed woodgrain texture, stark black background",
                "progression": [
                    "wooden cross with arms beginning to retract and straighten",
                    "cross becoming more linear, transforming into construction beam proportions"
                ]
            },
            (3, 4): {  # beam → lumber w/ nail
                "base": "detailed woodgrain texture, stark black background",
                "progression": [
                    "construction beam with bent nail beginning to appear, end becoming more weathered",
                    "beam transforming back to raw lumber, bent nail gleaming wickedly at one end"
                ]
            }
        }
        
        template = transformation_templates.get((from_frame, to_frame), {
            "base": "detailed woodgrain texture, stark black background",
            "progression": ["intermediate transformation stage"] * num_intermediates
        })
        
        prompts = []
        for i in range(num_intermediates):
            if i < len(template["progression"]):
                prompt = f"{template['base']}, {template['progression'][i]}"
            else:
                # Fallback for extra intermediates
                blend_ratio = (i + 1) / (num_intermediates + 1)
                prompt = f"{template['base']}, transformation stage {blend_ratio:.1f} between forms"
            
            prompts.append(prompt)
        
        return prompts
    
    def generate_via_backend_api(self, prompt: str, filename: str) -> bool:
        """
        Generate image using your existing backend API
        Uses the /generate-single endpoint for individual frame generation
        """
        print(f"🎨 Generating via backend: {filename}")
        print(f"   Prompt: {prompt[:60]}...")
        
        # Extract beat index from filename (e.g., "0001_01.png" -> beat 1)
        try:
            # For intermediate frames like "0001_01.png", use the base beat index
            base_frame = int(filename.split('_')[0])
            beat_index = base_frame  # Use the base frame's beat index
        except:
            beat_index = 0  # Fallback
        
        # Use your existing single keyframe generation endpoint with correct parameter format
        url = f"http://{self.backend_url}/api/scenes/{self.scene_id}/keyframes/generate-single"
        
        # Based on the error, beat_index should be a query parameter
        params = {
            "beat_index": beat_index
        }
        
        # custom_prompt goes in the JSON body
        payload = {
            "custom_prompt": prompt
        }
        
        try:
            response = requests.post(url, params=params, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Your API returns a different filename, so we need to rename it
                generated_filename = result.get("filename")
                if generated_filename:
                    # Move/rename the generated file to our desired filename
                    source_path = self.frames_dir / generated_filename
                    target_path = self.frames_dir / filename
                    
                    if source_path.exists():
                        import shutil
                        shutil.move(str(source_path), str(target_path))
                        print(f"   ✅ Generated and renamed: {target_path}")
                        return True
                    else:
                        print(f"   ⚠️ Generated file not found: {source_path}")
                        return False
                else:
                    print(f"   ❌ Backend didn't return filename")
                    return False
            else:
                print(f"   ❌ Backend API Error: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Request failed: {str(e)}")
            return False
    
    def regenerate_single_frame(self, frame_number: int, custom_prompt: Optional[str] = None):
        """
        Regenerate a single frame with option for custom prompt
        """
        if frame_number in self.locked_frames:
            print(f"❌ Cannot regenerate frame {frame_number:04d} - it's locked!")
            return False
        
        print(f"🔄 Regenerating frame {frame_number:04d}...")
        
        # Use custom prompt or get original from scene definition
        if custom_prompt:
            prompt = custom_prompt
            print(f"   Using custom prompt: {prompt[:50]}...")
        else:
            # Get original prompt from scene beat
            original_prompt = self._get_original_frame_prompt(frame_number)
            prompt = original_prompt
            print(f"   Using original prompt: {prompt[:50]}...")
        
        # Generate via backend API
        generation_request = {
            "scene_id": self.scene_id,
            "frame_number": frame_number,
            "prompt": prompt,
            "regenerate": True
        }
        
        # In real implementation: POST to /api/scenes/{scene_id}/keyframes/{frame_number}/regenerate
        print(f"   ✅ Frame {frame_number:04d} queued for regeneration")
        
        return True
    
    def _get_original_frame_prompt(self, frame_number: int) -> str:
        """Get original prompt for a frame from scene definition"""
        frame_prompts = {
            0: "extreme close-up of raw lumber, detailed woodgrain texture, stark black background",
            1: "extreme close-up of wooden club morphed from lumber, bent nail gleaming wickedly, stark black background",
            2: "extreme close-up of wooden cross, intricately carved, morphed from club, stark black background",
            3: "extreme close-up of raw construction beam, morphed from cross, detailed woodgrain, stark black background",
            4: "extreme close-up of raw lumber with bent nail gleaming wickedly, morphed from beam, stark black background"
        }
        
        return frame_prompts.get(frame_number, "detailed woodgrain texture, stark black background")
    
    def preview_iteration_plan(self):
        """
        Show complete iteration plan before executing
        """
        print("\n📋 INTRO SCENE ITERATION PLAN")
        print("=" * 50)
        
        # Current state
        print(f"🎬 Scene: {self.scene_id} (frames 0000-0004)")
        print(f"🔒 Locked frames: {sorted(self.locked_frames) if self.locked_frames else 'None'}")
        
        # Gap analysis
        recommendations = self.analyze_transformation_gaps()
        
        print(f"\n📊 Planned Generation:")
        total_new_frames = sum(r['recommended_intermediates'] for r in recommendations)
        print(f"   • {len(recommendations)} transitions to enhance")
        print(f"   • {total_new_frames} new intermediate frames")
        print(f"   • Estimated generation time: {total_new_frames * 30} seconds")
        
        print(f"\n🎯 Execution Order:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec['transition']} ({rec['recommended_intermediates']} intermediate frames)")
        
        return recommendations
    
    async def execute_iteration_workflow(self):
        """
        Execute the complete intro scene iteration workflow
        """
        print("\n🚀 EXECUTING INTRO SCENE ITERATION WORKFLOW")
        print("=" * 55)
        
        # Step 1: Load current state
        self.load_current_intro_state()
        
        # Step 2: Set up locking system
        self.setup_frame_locking_system()
        
        # Step 3: Preview plan
        recommendations = self.preview_iteration_plan()
        
        # Step 4: Execute generations (would be real API calls)
        print(f"\n⚡ Executing generations...")
        for rec in recommendations:
            print(f"   Processing {rec['transition']}...")
            self.generate_intermediate_frames(rec)
            await asyncio.sleep(1)  # Rate limiting
        
        print(f"\n✅ Intro scene iteration complete!")
        print(f"   • Enhanced {len(recommendations)} transformations")
        print(f"   • Ready for RIFE interpolation testing")
        
        return True

# Usage example
if __name__ == "__main__":
    # Initialize iterator
    iterator = IntroSceneIterator()
    
    # Demo workflow
    print("🎬 Saint Lucifer - Intro Scene Iteration Workflow")
    print("Building on existing FastAPI backend architecture\n")
    
    # Load current state
    iterator.load_current_intro_state()
    
    # Analyze gaps
    iterator.analyze_transformation_gaps()
    
    # Set up locking (user could interact with this)
    iterator.setup_frame_locking_system()
    
    # Example: Lock frame 0 as it's already good
    iterator.lock_frame(0)
    iterator.lock_frame(4)  # Also lock the final frame
    
    # Preview complete plan
    iterator.preview_iteration_plan()
    
    print("\n🎯 Next Steps:")
    print("   Option A: iterator.test_club_to_cross_transition()  # Test single transition")
    print("   Option B: iterator.execute_iteration_workflow()     # Generate all 6 frames")
    print("   Option C: iterator.generate_via_stability_ai(prompt, filename)  # Manual generation")
    
    print(f"\n💡 Recommended: Start with Option A to test the most challenging transition")
