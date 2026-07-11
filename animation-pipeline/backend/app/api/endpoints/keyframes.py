from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from crud.keyframes import KeyframeCRUD
from crud.beats import BeatCRUD
from datetime import datetime
import os
import base64
import requests
import time
from pathlib import Path
import sqlite3

router = APIRouter()


# Pydantic models
class KeyframeSetCreate(BaseModel):
    scene_id: str
    generation_params: Dict[str, Any] = {
        "api_provider": "stability",
        "model": "stable-diffusion-xl-1024-v1-0",
        "resolution": "1024x1024",
        "steps": 30,
        "cfg_scale": 7
    }
    version: Optional[int] = None


class GeneratedFrameCreate(BaseModel):
    keyframe_set_id: int
    filename: str
    beat_index: int
    prompt_used: str
    seed: Optional[int] = None
    metadata: Dict[str, Any] = {}


class KeyframeSetResponse(BaseModel):
    id: int
    scene_id: str
    version: int
    generation_params: Dict[str, Any]
    created_at: str

    class Config:
        from_attributes = True


class GeneratedFrameResponse(BaseModel):
    id: int
    keyframe_set_id: int
    filename: str
    beat_index: int
    prompt_used: str
    seed: Optional[int]
    metadata: Dict[str, Any]

    class Config:
        from_attributes = True

class StabilityAIGenerator:
    """Real Stability AI image generation"""
    
    def __init__(self):
        self.api_key = os.getenv("STABILITY_API_KEY")
        self.session = requests.Session()
        # Directory where images will be saved (same as your image serving directory)
        self.output_dir = Path.home() / "Development" / "layersofmeta" / "frames"
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_image(self, prompt: str, filename: str) -> bool:
        """Generate image using Stability AI API"""
        if not self.api_key:
            print("❌ Stability API key not configured")
            return False
        
        url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
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
            print(f"🎨 Generating: {filename}")
            print(f"📝 Prompt: {prompt[:100]}...")
            
            response = self.session.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            # Save image to frames directory (where your images are served from)
            image_data = base64.b64decode(data["artifacts"][0]["base64"])
            output_path = self.output_dir / filename
            
            with open(output_path, "wb") as f:
                f.write(image_data)
            
            print(f"✅ Generated: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error generating {filename}: {e}")
            return False
            
# Create global generator instance
stability_generator = StabilityAIGenerator()

# Keyframe Set Management
@router.post("/keyframes/sets/", response_model=KeyframeSetResponse)
async def create_keyframe_set(keyframe_set_data: KeyframeSetCreate):
    """Create a new keyframe set for a scene"""
    try:
        keyframe_set = KeyframeCRUD.create_keyframe_set(
            scene_id=keyframe_set_data.scene_id,
            generation_params=keyframe_set_data.generation_params,
            version=keyframe_set_data.version
        )
        
        # Convert datetime to string for response
        response_data = {
            "id": keyframe_set.id,
            "scene_id": keyframe_set.scene_id,
            "version": keyframe_set.version,
            "generation_params": keyframe_set.generation_params,
            "created_at": keyframe_set.created_at.isoformat()
        }
        
        return response_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/keyframes/sets/{keyframe_set_id}", response_model=KeyframeSetResponse)
async def get_keyframe_set(keyframe_set_id: int):
    """Get a specific keyframe set"""
    keyframe_set = KeyframeCRUD.get_keyframe_set_by_id(keyframe_set_id)
    if not keyframe_set:
        raise HTTPException(status_code=404, detail="Keyframe set not found")
    
    return {
        "id": keyframe_set.id,
        "scene_id": keyframe_set.scene_id,
        "version": keyframe_set.version,
        "generation_params": keyframe_set.generation_params,
        "created_at": keyframe_set.created_at.isoformat()
    }


@router.get("/scenes/{scene_id}/keyframes/sets")
async def get_scene_keyframe_sets(scene_id: str):
    """Get all keyframe sets for a scene"""
    keyframe_sets = KeyframeCRUD.get_keyframe_sets_for_scene(scene_id)
    
    return [
        {
            "id": ks.id,
            "scene_id": ks.scene_id,
            "version": ks.version,
            "generation_params": ks.generation_params,
            "created_at": ks.created_at.isoformat()
        }
        for ks in keyframe_sets
    ]


@router.delete("/keyframes/sets/{keyframe_set_id}")
async def delete_keyframe_set(keyframe_set_id: int):
    """Delete a keyframe set and all its frames"""
    success = KeyframeCRUD.delete_keyframe_set(keyframe_set_id)
    if not success:
        raise HTTPException(status_code=404, detail="Keyframe set not found")
    return {"message": "Keyframe set deleted successfully"}


# Generated Frame Management
@router.post("/keyframes/frames/", response_model=GeneratedFrameResponse)
async def add_generated_frame(frame_data: GeneratedFrameCreate):
    """Add a generated frame to a keyframe set"""
    try:
        frame = KeyframeCRUD.add_generated_frame(
            keyframe_set_id=frame_data.keyframe_set_id,
            filename=frame_data.filename,
            beat_index=frame_data.beat_index,
            prompt_used=frame_data.prompt_used,
            seed=frame_data.seed,
            metadata=frame_data.metadata
        )
        return frame
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/keyframes/sets/{keyframe_set_id}/frames")
async def get_keyframe_set_frames(keyframe_set_id: int):
    """Get all frames for a keyframe set"""
    frames = KeyframeCRUD.get_frames_for_keyframe_set(keyframe_set_id)
    return frames


# Scene Generation Status and Control
@router.get("/scenes/{scene_id}/generation/status")
async def get_scene_generation_status(scene_id: str):
    """Get complete generation status for a scene"""
    status = KeyframeCRUD.get_scene_generation_status(scene_id)
    if not status:
        raise HTTPException(status_code=404, detail="Scene not found")
    return status


@router.post("/scenes/{scene_id}/generation/prepare")
async def prepare_scene_generation(scene_id: str, generation_params: Dict[str, Any] = None):
    """Prepare a scene for keyframe generation by creating keyframe set"""
    
    # Check if scene has beats
    beats = BeatCRUD.get_beats_for_scene(scene_id)
    if not beats:
        raise HTTPException(
            status_code=400,
            detail="Scene must have beats defined before generating keyframes"
        )
    
    # Set default generation parameters
    if not generation_params:
        generation_params = {
            "api_provider": "stability",
            "model": "stable-diffusion-xl-1024-v1-0",
            "resolution": "1024x1024",
            "steps": 30,
            "cfg_scale": 7
        }
    
    try:
        keyframe_set = KeyframeCRUD.create_keyframe_set(scene_id, generation_params)
        
        return {
            "message": "Scene prepared for generation",
            "keyframe_set_id": keyframe_set.id,
            "total_beats": len(beats),
            "generation_params": generation_params
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Bulk Operations for Generation Pipeline
@router.post("/scenes/{scene_id}/generation/simulate")
async def simulate_scene_generation(scene_id: str):
    """Simulate keyframe generation for testing (creates placeholder frames)"""
    
    # Get the latest keyframe set or create one
    latest_set = KeyframeCRUD.get_latest_keyframe_set(scene_id)
    if not latest_set:
        # Create a keyframe set first
        latest_set = KeyframeCRUD.create_keyframe_set(scene_id, {
            "api_provider": "simulation",
            "model": "test"
        })
    
    # Get beats for the scene
    beats = BeatCRUD.get_beats_for_scene(scene_id)
    if not beats:
        raise HTTPException(status_code=400, detail="Scene must have beats defined")
    
    generated_frames = []
    
    try:
        for beat in beats:
            # Create a simulated frame
            filename = f"frame_{scene_id}_{beat.beat_order:03d}_simulated.png"
            prompt = f"Simulated prompt for beat: {beat.description}"
            
            frame = KeyframeCRUD.add_generated_frame(
                keyframe_set_id=latest_set.id,
                filename=filename,
                beat_index=beat.beat_order,
                prompt_used=prompt,
                metadata={"simulated": True, "beat_id": beat.id}
            )
            
            generated_frames.append({
                "frame_id": frame.id,
                "filename": filename,
                "beat_description": beat.description
            })
        
        return {
            "message": "Scene generation simulated successfully",
            "keyframe_set_id": latest_set.id,
            "generated_frames": generated_frames,
            "total_frames": len(generated_frames)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Analysis and Reporting
@router.get("/scenes/{scene_id}/generation/summary")
async def get_scene_generation_summary(scene_id: str):
    """Get a summary of generation progress and statistics"""
    status = KeyframeCRUD.get_scene_generation_status(scene_id)
    if not status:
        raise HTTPException(status_code=404, detail="Scene not found")
    
    # Calculate additional statistics
    beats = status["beats"]
    frames = status["generated_frames"]
    
    # Find which beats have generated frames
    generated_beat_indices = {frame["beat_index"] for frame in frames}
    missing_beats = [beat for beat in beats if beat["beat_order"] not in generated_beat_indices]
    
    return {
        "scene_info": status["scene"],
        "progress": status["progress"],
        "keyframe_set": status["keyframe_set"],
        "missing_beats": missing_beats,
        "recent_frames": frames[-5:] if len(frames) > 5 else frames,  # Last 5 frames
        "generation_ready": len(missing_beats) == 0
    }


# Frontend-compatible endpoints
@router.get("/scenes/{scene_id}/beats/timeline")
async def get_scene_timeline_for_frontend(scene_id: str):
    """
    Get scene timeline data in the format expected by the React component
    This endpoint matches what SceneTimeline.js is calling
    """
    from crud.scenes import SceneCRUD
    
    # Get scene data
    scene = SceneCRUD.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    
    # Get beats for the scene
    beats = BeatCRUD.get_beats_for_scene(scene_id)
    
    return {
        "scene": {
            "id": scene.id,
            "name": scene.name,
            "duration": scene.duration,
            "start_time": scene.start_time,
            "end_time": scene.start_time + scene.duration,
            "narrative_prompt": scene.narrative_prompt
        },
        "beats": [
            {
                "id": beat.id,
                "timestamp": beat.timestamp,
                "description": beat.description,
                "characters": beat.characters,
                "beat_order": beat.beat_order
            }
            for beat in beats
        ]
    }

# Replace your existing keyframe generation endpoint with this real one:
@router.post("/scenes/{scene_id}/keyframes/generate")
async def start_real_keyframe_generation(scene_id: str, generation_params: Dict[str, Any] = None):
    """
    REAL keyframe generation using Stability AI
    This replaces the simulation with actual AI generation
    """
    # Check if scene exists
    from crud.scenes import SceneCRUD
    scene = SceneCRUD.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    
    # Check if scene has beats
    beats = BeatCRUD.get_beats_for_scene(scene_id)
    if not beats:
        raise HTTPException(
            status_code=400,
            detail="Scene must have beats defined before generating keyframes"
        )
    
    # Check API key
    if not stability_generator.api_key:
        raise HTTPException(
            status_code=400,
            detail="Stability AI API key not configured. Add STABILITY_API_KEY to your environment."
        )
    
    # Set generation parameters
    if not generation_params:
        generation_params = {
            "api_provider": "stability_ai",
            "model": "stable-diffusion-xl-1024-v1-0",
            "resolution": "1024x1024",
            "steps": 30,
            "cfg_scale": 7,
            "style_preset": "film noir chiaroscuro lighting, wet pavement reflections"
        }
    
    try:
        # Create a new keyframe set for this generation
        keyframe_set = KeyframeCRUD.create_keyframe_set(scene_id, generation_params)
        
        # Build style context for this scene
        scene_style = scene.style_config.get("style", "") if scene.style_config else ""
        base_style = generation_params.get("style_preset", "cinematic lighting")
        
        generated_frames = []
        failed_frames = []
        
        print(f"\n🎬 Starting REAL generation for scene '{scene_id}' with {len(beats)} beats")
        
        for i, beat in enumerate(beats):
            # Create unique filename for each generated frame
            filename = f"gen_{scene_id}_{beat.beat_order:03d}_{keyframe_set.id}.png"
            
            # Build comprehensive prompt
            prompt_parts = [
                beat.description,
                scene_style,
                base_style,
                "high quality, cinematic composition, dramatic lighting"
            ]
            
            # Clean and combine prompt parts
            full_prompt = ", ".join([part.strip() for part in prompt_parts if part.strip()])
            
            print(f"\n🎨 Generating beat {i+1}/{len(beats)}: {beat.description[:50]}...")
            
            # Generate with Stability AI
            success = stability_generator.generate_image(full_prompt, filename)
            
            if success:
                # Add to database
                frame = KeyframeCRUD.add_generated_frame(
                    keyframe_set_id=keyframe_set.id,
                    filename=filename,
                    beat_index=beat.beat_order,
                    prompt_used=full_prompt,
                    metadata={
                        "beat_id": beat.id,
                        "generation_type": "stability_ai",
                        "timestamp": beat.timestamp,
                        "scene_style": scene_style,
                        "api_response": "success"
                    }
                )
                
                generated_frames.append({
                    "frame_id": frame.id,
                    "filename": filename,
                    "beat_description": beat.description,
                    "prompt": full_prompt
                })
            else:
                failed_frames.append({
                    "beat_order": beat.beat_order,
                    "description": beat.description,
                    "error": "API generation failed"
                })
            
            # Rate limiting - be respectful to API
            if i < len(beats) - 1:  # Don't sleep after last frame
                print("⏱️ Rate limiting (2 seconds)...")
                time.sleep(2)
        
        success_count = len(generated_frames)
        total_count = len(beats)
        
        return {
            "message": f"Real keyframe generation completed: {success_count}/{total_count} successful",
            "keyframe_set_id": keyframe_set.id,
            "scene_id": scene_id,
            "total_beats": total_count,
            "successful_frames": success_count,
            "failed_frames": len(failed_frames),
            "generated_frames": generated_frames,
            "failed_details": failed_frames,
            "status": "completed" if success_count == total_count else "partial",
            "generation_params": generation_params
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Real generation failed: {str(e)}")

@router.get("/scenes/{scene_id}/keyframes/status")
async def get_keyframe_generation_status(scene_id: str):
    """
    Get the current keyframe generation status for a scene
    Returns progress information for the UI
    """
    try:
        # Check if scene exists first
        from crud.scenes import SceneCRUD
        scene = SceneCRUD.get_scene(scene_id)
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")
        
        # Get beats count
        beats = BeatCRUD.get_beats_for_scene(scene_id)
        total_beats = len(beats)
        
        # Get latest keyframe set
        latest_keyframe_set = KeyframeCRUD.get_latest_keyframe_set(scene_id)
        
        if not latest_keyframe_set:
            return {
                "scene_id": scene_id,
                "progress": {
                    "total_beats": total_beats,
                    "generated_frames": 0,
                    "percentage": 0,
                    "status": "not_started"
                },
                "keyframe_set": None,
                "generated_frames": [],
                "status": "not_started",
                "message": "No keyframes generated yet"
            }
        
        # Get generated frames count
        frames = KeyframeCRUD.get_frames_for_keyframe_set(latest_keyframe_set.id)
        generated_count = len(frames)
        percentage = (generated_count / total_beats * 100) if total_beats > 0 else 0
        
        return {
            "scene_id": scene_id,
            "progress": {
                "total_beats": total_beats,
                "generated_frames": generated_count,
                "percentage": round(percentage, 2),
                "status": "complete" if generated_count == total_beats else "in_progress" if generated_count > 0 else "not_started"
            },
            "keyframe_set": {
                "id": latest_keyframe_set.id,
                "version": latest_keyframe_set.version,
                "created_at": latest_keyframe_set.created_at.isoformat()
            },
            "generated_frames": [
                {
                    "id": frame.id,
                    "filename": frame.filename,
                    "beat_index": frame.beat_index
                }
                for frame in frames
            ],
            "status": "complete" if generated_count == total_beats else "in_progress",
            "message": f"Generated {generated_count} of {total_beats} keyframes"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.get("/scenes/{scene_id}/keyframes")
async def get_scene_keyframes(scene_id: str):
    """
    Get all keyframes for a scene (latest version) - simplified version
    """
    try:
        # Check if scene exists
        from crud.scenes import SceneCRUD
        scene = SceneCRUD.get_scene(scene_id)
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")
        
        latest_keyframe_set = KeyframeCRUD.get_latest_keyframe_set(scene_id)
        
        if not latest_keyframe_set:
            return {
                "scene_id": scene_id,
                "keyframes": [],
                "total_count": 0,
                "message": "No keyframes generated yet"
            }
        
        frames = KeyframeCRUD.get_frames_for_keyframe_set(latest_keyframe_set.id)
        
        return {
            "scene_id": scene_id,
            "keyframe_set_id": latest_keyframe_set.id,
            "keyframe_set_version": latest_keyframe_set.version,
            "keyframes": [
                {
                    "id": frame.id,
                    "filename": frame.filename,
                    "beat_index": frame.beat_index,
                    "prompt_used": frame.prompt_used
                }
                for frame in frames
            ],
            "total_count": len(frames)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get keyframes: {str(e)}")


# MISSING IMPORT ENDPOINT - This is what was causing the 404!
@router.post("/scenes/{scene_id}/keyframes/import")
async def import_existing_keyframes(scene_id: str, keyframes_data: List[Dict[str, Any]]):
    """
    Import keyframes from your existing pipeline
    Useful for integrating your 44 existing keyframes
    """
    from crud.scenes import SceneCRUD
    
    # Verify scene exists
    scene = SceneCRUD.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    
    try:
        # Create a keyframe set for imported frames
        keyframe_set = KeyframeCRUD.create_keyframe_set(
            scene_id=scene_id,
            generation_params={
                "api_provider": "imported",
                "source": "existing_pipeline",
                "import_timestamp": datetime.utcnow().isoformat()
            }
        )
        
        imported_frames = []
        
        for frame_data in keyframes_data:
            frame = KeyframeCRUD.add_generated_frame(
                keyframe_set_id=keyframe_set.id,
                filename=frame_data["filename"],
                beat_index=frame_data.get("beat_index", 0),
                prompt_used=frame_data.get("prompt", "Imported frame"),
                seed=frame_data.get("seed"),
                metadata=frame_data.get("metadata", {"imported": True})
            )
            imported_frames.append(frame)
        
        return {
            "message": f"Successfully imported {len(imported_frames)} keyframes",
            "keyframe_set_id": keyframe_set.id,
            "scene_id": scene_id,
            "imported_count": len(imported_frames)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


# Health check for keyframe generation system
@router.get("/keyframes/health")
async def keyframes_health_check():
    """Check if keyframe generation system is healthy"""
    try:
        # Test database connection
        from database import get_db
        db = next(get_db())
        
        # Count total keyframe sets
        from models.scenes import KeyframeSet, GeneratedFrame
        total_sets = db.query(KeyframeSet).count()
        total_frames = db.query(GeneratedFrame).count()
        
        db.close()
        
        return {
            "status": "healthy",
            "total_keyframe_sets": total_sets,
            "total_generated_frames": total_frames,
            "endpoints_available": [
                "POST /api/scenes/{scene_id}/keyframes/generate",
                "GET /api/scenes/{scene_id}/keyframes/status",
                "GET /api/scenes/{scene_id}/keyframes",
                "POST /api/scenes/{scene_id}/keyframes/import"
            ]
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Add a new endpoint to test individual frame generation
@router.post("/scenes/{scene_id}/keyframes/generate-single")
async def generate_single_keyframe(scene_id: str, beat_index: int, custom_prompt: str = None):
    """
    Generate a single keyframe for testing
    Useful for experimenting with prompts before full scene generation
    """
    from crud.scenes import SceneCRUD
    
    # Check scene exists
    scene = SceneCRUD.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    
    # Get the specific beat
    beats = BeatCRUD.get_beats_for_scene(scene_id)
    target_beat = None
    for beat in beats:
        if beat.beat_order == beat_index:
            target_beat = beat
            break
    
    if not target_beat:
        raise HTTPException(status_code=404, detail=f"Beat {beat_index} not found in scene")
    
    # Check API key
    if not stability_generator.api_key:
        raise HTTPException(
            status_code=400,
            detail="Stability AI API key not configured"
        )
    
    try:
        # Create keyframe set for single generation
        keyframe_set = KeyframeCRUD.create_keyframe_set(scene_id, {
            "api_provider": "stability_ai",
            "generation_type": "single_test"
        })
        
        # Use custom prompt or build from beat
        if custom_prompt:
            full_prompt = custom_prompt
        else:
            scene_style = scene.style_config.get("style", "") if scene.style_config else ""
            full_prompt = f"{target_beat.description}, {scene_style}, cinematic lighting"
        
        # Generate single frame
        filename = f"test_{scene_id}_{beat_index}_{keyframe_set.id}.png"
        
        success = stability_generator.generate_image(full_prompt, filename)
        
        if success:
            frame = KeyframeCRUD.add_generated_frame(
                keyframe_set_id=keyframe_set.id,
                filename=filename,
                beat_index=beat_index,
                prompt_used=full_prompt,
                metadata={"test_generation": True}
            )
            
            return {
                "message": "Single keyframe generated successfully",
                "filename": filename,
                "prompt_used": full_prompt,
                "beat_description": target_beat.description,
                "frame_id": frame.id,
                "image_url": f"/images/{filename}"
            }
        else:
            raise HTTPException(status_code=500, detail="Generation failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Single generation failed: {str(e)}")


# Add endpoint to check API key status
@router.get("/keyframes/stability-status")
async def check_stability_status():
    """Check if Stability AI is properly configured"""
    has_key = bool(stability_generator.api_key)
    output_dir_exists = stability_generator.output_dir.exists()
    
    return {
        "api_key_configured": has_key,
        "output_directory": str(stability_generator.output_dir),
        "output_directory_exists": output_dir_exists,
        "ready_for_generation": has_key and output_dir_exists,
        "status": "ready" if (has_key and output_dir_exists) else "needs_configuration"
    }
        
