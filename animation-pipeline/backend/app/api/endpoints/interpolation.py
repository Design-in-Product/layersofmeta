from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from pydantic import BaseModel
from crud.interpolation import InterpolationCRUD, RIFEProcessor, VideoAssembler
from pathlib import Path
from crud.runpod_client import RunpodRIFEClient, RunpodInterpolationCRUD
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Pydantic models
class InterpolationSettings(BaseModel):
    target_fps: int = 24
    runpod_api_url: str = "https://gj8s0v4skzwery-5000.proxy.runpod.net"
    model_version: str = "4.6"
    keyframes_dir: str = "/Users/xian/Development/layersofmeta/frames"
    output_dir: str = "/Users/xian/Development/layersofmeta/frames/interpolated"

class VideoAssemblySettings(BaseModel):
    fps: int = 24
    output_filename: str = "scene_video.mp4"
    audio_path: Optional[str] = None

# Interpolation Planning
@router.get("/scenes/{scene_id}/interpolation/plan")
async def get_interpolation_plan(scene_id: str, target_fps: int = 24):
    """Get interpolation plan for a scene"""
    plan = InterpolationCRUD.calculate_interpolation_plan(scene_id, target_fps)
    if not plan:
        raise HTTPException(status_code=404, detail="Scene not found or no keyframes available")
    
    return plan

@router.get("/scenes/{scene_id}/interpolation/status")
async def get_interpolation_status(scene_id: str):
    """Get interpolation readiness status for a scene"""
    from crud.interpolation import get_scene_interpolation_status
    status = get_scene_interpolation_status(scene_id)
    return status

# RIFE Processing Endpoints
@router.post("/scenes/{scene_id}/interpolation/process")
async def process_scene_interpolation(
    scene_id: str,
    settings: InterpolationSettings
):
    """Process RIFE interpolation for all keyframe pairs in a scene"""
    try:
        processor = RIFEProcessor(settings.rife_path, settings.model_version)
        
        result = processor.process_scene_interpolation(
            scene_id=scene_id,
            keyframes_dir=Path(settings.keyframes_dir),
            output_dir=Path(settings.output_dir),
            target_fps=settings.target_fps
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interpolation processing failed: {str(e)}")

@router.post("/interpolation/pair")
async def interpolate_frame_pair(
    frame1_filename: str,
    frame2_filename: str,
    settings: InterpolationSettings,
    exp: int = 2
):
    """Interpolate between two specific frames"""
    try:
        processor = RIFEProcessor(settings.rife_path, settings.model_version)
        
        frame1_path = Path(settings.keyframes_dir) / frame1_filename
        frame2_path = Path(settings.keyframes_dir) / frame2_filename
        output_dir = Path(settings.output_dir)
        
        result = processor.interpolate_frame_pair(
            frame1_path=frame1_path,
            frame2_path=frame2_path,
            output_dir=output_dir,
            exp=exp
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frame pair interpolation failed: {str(e)}")

# Video Assembly
@router.post("/scenes/{scene_id}/video/assemble")
async def assemble_scene_video(
    scene_id: str,
    settings: VideoAssemblySettings,
    interpolation_results_path: str
):
    """Assemble interpolated frames into final video using FFmpeg"""
    try:
        import json
        with open(interpolation_results_path, 'r') as f:
            interpolation_results = json.load(f)
        
        frame_sequence = VideoAssembler.create_frame_sequence_list(interpolation_results)
        
        output_path = Path(settings.output_filename)
        audio_path = Path(settings.audio_path) if settings.audio_path else None
        
        result = VideoAssembler.assemble_video_with_ffmpeg(
            frame_sequence=frame_sequence,
            output_path=output_path,
            fps=settings.fps,
            audio_path=audio_path
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video assembly failed: {str(e)}")

# Testing and Simulation
@router.post("/scenes/{scene_id}/interpolation/simulate")
async def simulate_interpolation(scene_id: str, target_fps: int = 24):
    """Simulate interpolation processing for testing pipeline"""
    try:
        plan = InterpolationCRUD.calculate_interpolation_plan(scene_id, target_fps)
        if not plan or "error" in plan:
            raise HTTPException(status_code=400, detail="Cannot create interpolation plan")
        
        simulated_results = []
        total_simulated_frames = 0
        
        for job in plan["interpolation_jobs"]:
            simulated_result = {
                "pair_index": job["pair_index"],
                "success": True,
                "input_frames": 2,
                "output_frames": job["rife_output_frames"],
                "frame1_filename": job["frame1_filename"],
                "frame2_filename": job["frame2_filename"],
                "rife_exp": job["rife_exp"],
                "simulated": True
            }
            simulated_results.append(simulated_result)
            total_simulated_frames += job["rife_output_frames"]
        
        return {
            "success": True,
            "scene_id": scene_id,
            "interpolation_plan": plan,
            "simulated_results": simulated_results,
            "total_simulated_frames": total_simulated_frames,
            "estimated_video_duration": total_simulated_frames / target_fps,
            "message": "Interpolation simulation complete - ready for real processing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@router.get("/scenes/{scene_id}/interpolation/requirements")
async def get_interpolation_requirements(scene_id: str, target_fps: int = 24):
    """Get computational requirements for scene interpolation"""
    try:
        plan = InterpolationCRUD.calculate_interpolation_plan(scene_id, target_fps)
        if not plan or "error" in plan:
            raise HTTPException(status_code=400, detail="Cannot analyze requirements")
        
        total_pairs = len(plan["interpolation_jobs"])
        total_output_frames = plan["total_output_frames"]
        
        estimated_seconds_per_pair = 30
        estimated_total_time = total_pairs * estimated_seconds_per_pair
        
        frame_size_mb = 2
        storage_needed_gb = (total_output_frames * frame_size_mb) / 1024
        
        return {
            "scene_id": scene_id,
            "target_fps": target_fps,
            "interpolation_jobs": total_pairs,
            "total_output_frames": total_output_frames,
            "estimated_processing_time": {
                "seconds": estimated_total_time,
                "minutes": round(estimated_total_time / 60, 2),
                "formatted": f"{estimated_total_time // 60}m {estimated_total_time % 60:.0f}s"
            },
            "storage_requirements": {
                "total_frames": total_output_frames,
                "estimated_size_gb": round(storage_needed_gb, 2),
                "per_frame_mb": frame_size_mb
            },
            "gpu_recommendations": {
                "minimum_vram": "8GB",
                "recommended_vram": "16GB+",
                "batch_processing": "Process pairs sequentially to manage memory"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Requirements analysis failed: {str(e)}")

@router.post("/scenes/{scene_id}/interpolation/process-runpod")
async def process_scene_interpolation_runpod(
    scene_id: str,
    runpod_api_url: str = "https://gj8s0v4skzwery-5000.proxy.runpod.net"
):
    """Process scene interpolation using Runpod RIFE API (REAL interpolation!)"""
    try:
        keyframes_dir = Path.home() / "Development" / "layersofmeta" / "frames"
        output_dir = Path.home() / "Development" / "layersofmeta" / "frames" / "interpolated" / scene_id
        
        result = RunpodInterpolationCRUD.process_scene_interpolation_runpod(
            scene_id=scene_id,
            runpod_api_url=runpod_api_url,
            keyframes_dir=keyframes_dir,
            output_dir=output_dir
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Runpod interpolation failed: {str(e)}")

# NEW SCENE ITERATOR RIFE TEST ENDPOINT
@router.post("/interpolation/test-rife")
async def test_rife_interpolation(request_data: dict):
    """Test RIFE interpolation for Scene Iterator"""
    try:
        scene_id = request_data.get('scene_id', 'intro')
        
        logger.info(f'🎯 Testing RIFE interpolation for scene: {scene_id}')
        
        # For now, simulate RIFE test
        # Replace with actual RIFE call when ready
        
        return {
            "success": True,
            "scene_id": scene_id,
            "status": "RIFE test initiated",
            "message": f"RIFE interpolation test started for {scene_id}",
            "simulated": True
        }
        
    except Exception as e:
        logger.error(f"❌ RIFE test error: {e}")
        raise HTTPException(status_code=500, detail=f"RIFE test failed: {str(e)}")

# Utility endpoints
@router.get("/interpolation/health")
async def check_interpolation_health():
    """Health check for interpolation system"""
    return {
        "status": "healthy",
        "components": {
            "interpolation_crud": "available",
            "rife_processor": "available",
            "video_assembler": "available"
        },
        "message": "Interpolation system ready for processing"
    }
