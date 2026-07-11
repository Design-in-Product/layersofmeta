"""
Main FastAPI application for Animation Pipeline 2.0
Complete backend/app/main.py file with scene management integration
"""
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager
from database import init_database, db_config
from api.endpoints.beats import router as beats_router
from api.endpoints.scenes import router as scenes_router
from api.endpoints.keyframes import router as keyframes_router
from api.endpoints.interpolation import router as interpolation_router
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path
from fastapi.responses import FileResponse
import base64
import requests
import json
from typing import Dict, Any
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Any, Optional
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional

class GenerateFrameRequest(BaseModel):
    frame: int
    prompt: str
    scene_id: str = "intro"

class SavePromptRequest(BaseModel):
    frame: int
    prompt: str
    version_filename: str = None

class GenerateFrameResponse(BaseModel):
    success: bool
    filename: str = None
    error: str = None

class SavePromptResponse(BaseModel):
    success: bool
    error: str = None

class SceneBeatRequest(BaseModel):
    scene_id: str
    beat_id: int
    prompt: str
    description: str = ""
    
class SceneBeatResponse(BaseModel):
    success: bool
    beat_id: int = None
    filename: str = None
    error: str = None

class LockFrameRequest(BaseModel):
    scene_id: str
    beat_id: int
    locked: bool
    
    

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown tasks
    """
    # Startup
    logger.info("Starting Animation Pipeline API...")
    try:
        # Initialize database
        init_database()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Animation Pipeline API...")

# Create FastAPI application
app = FastAPI(
    title="Animation Pipeline 2.0 API",
    description="AI-assisted music video creation pipeline with scene-based management",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# This serves your keyframe images at /images/
keyframes_dir = Path.home() / "Development" / "layersofmeta" / "frames"
if keyframes_dir.exists():
    app.mount("/images", StaticFiles(directory=str(keyframes_dir)), name="images")
    logger.info(f"✅ Serving keyframes from {keyframes_dir}")
else:
    logger.warning(f"⚠️ Keyframes directory not found: {keyframes_dir}")

@app.get("/simple_generator.html")
async def serve_simple_generator():
    """Serve the Simple Frame Generator UI"""
    generator_path = Path("/Users/xian/Development/layersofmeta/simple_generator.html")
    if generator_path.exists():
        return FileResponse(generator_path)
    else:
        raise HTTPException(status_code=404, detail="Simple Generator not found")
        
@app.get("/keyframe_refinery.html")
async def serve_refinery():
    # Adjust path to where your keyframe_refinery.html is located
    refinery_path = Path("/Users/xian/Development/layersofmeta/keyframe_refinery.html")
    if refinery_path.exists():
        return FileResponse(refinery_path)
    else:
        raise HTTPException(status_code=404, detail="Refinery not found")

@app.get("/scene_iterator.html")
async def serve_scene_iterator():
    """Serve the Scene Iterator UI"""
    iterator_path = Path("/Users/xian/Development/layersofmeta/scene_iterator.html")
    if iterator_path.exists():
        return FileResponse(iterator_path)
    else:
        raise HTTPException(status_code=404, detail="Scene Iterator not found")

# Include API routers
app.include_router(scenes_router)
app.include_router(beats_router, prefix="/api")
app.include_router(keyframes_router, prefix="/api")
app.include_router(interpolation_router, prefix="/api")

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint with basic information"""
    return {
        "message": "Animation Pipeline 2.0 API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Application health check
    Tests database connectivity and returns system status
    """
    try:
        # Test database connection
        db_healthy = db_config.test_connection()
        
        return {
            "status": "healthy" if db_healthy else "unhealthy",
            "database": "connected" if db_healthy else "disconnected",
            "version": "2.0.0"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "version": "2.0.0"
            }
        )

# ADD THESE ENDPOINTS AFTER YOUR health_check ENDPOINT AND BEFORE global_exception_handler
# Insert after line 122 (after your health check function) and before the global exception handler

@app.get("/api/frame/{frame_id}")
async def get_frame_image(frame_id: str):
    """
    Get frame image by ID - serves images from the frames directory
    This bridges the gap between refinery's expected /api/frame/ and our /images/ setup
    """
    try:
        # Convert frame ID to filename (e.g., "0001" -> "0001.png")
        if not frame_id.endswith('.png'):
            filename = f"{frame_id}.png"
        else:
            filename = frame_id
            
        # Check if file exists in keyframes directory
        image_path = keyframes_dir / filename
        
        if image_path.exists():
            return FileResponse(
                image_path,
                media_type="image/png",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            # Return 404 if image doesn't exist
            raise HTTPException(status_code=404, detail=f"Frame {frame_id} not found")
            
    except Exception as e:
        logger.error(f"Error serving frame {frame_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading frame: {str(e)}")

@app.get("/api/frame/original/{frame_id}")
async def get_original_frame(frame_id: str):
    """
    Get original version of a frame (fallback to regular frame endpoint)
    """
    return await get_frame_image(frame_id)

@app.post("/api/generate")
async def generate_frame(request: GenerateFrameRequest):
    """
    Generate new keyframe using Stability AI
    Called by refinery when user clicks "Generate New"
    """
    try:
        logger.info(f"🎨 Generating frame {request.frame:04d} with prompt: {request.prompt[:50]}...")
        
        # Get Stability AI API key
        stability_api_key = os.getenv("STABILITY_API_KEY")
        
        if not stability_api_key:
            return GenerateFrameResponse(
                success=False,
                error="Stability AI API key not configured"
            )
        
        # Prepare API request
        url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
        
        headers = {
            "Authorization": f"Bearer {stability_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "text_prompts": [{"text": request.prompt, "weight": 1.0}],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 30
        }
        
        # Make API call
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            # Generate filename with timestamp to avoid conflicts
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"{request.frame:04d}_v{timestamp}.png"
            
            # Save the generated image
            for artifact in data["artifacts"]:
                image_data = base64.b64decode(artifact["base64"])
                output_path = keyframes_dir / filename
                
                with open(output_path, "wb") as f:
                    f.write(image_data)
                
                logger.info(f"✅ Generated and saved: {filename}")
                
                return GenerateFrameResponse(
                    success=True,
                    filename=filename
                )
        else:
            error_msg = f"Stability AI API error: {response.status_code}"
            logger.error(f"❌ {error_msg}")
            return GenerateFrameResponse(
                success=False,
                error=error_msg
            )
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return GenerateFrameResponse(
            success=False,
            error=error_msg
        )
    except Exception as e:
        error_msg = f"Generation error: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return GenerateFrameResponse(
            success=False,
            error=error_msg
        )

@app.post("/api/prompts/save")
async def save_refined_prompt(request: SavePromptRequest):
    """
    Save refined prompt for a frame
    Called when user approves a prompt in the refinery
    """
    try:
        # Define path for refined prompts JSON file
        refined_prompts_path = keyframes_dir.parent / "refined_prompts.json"
        
        # Load existing refined prompts or create new dict
        if refined_prompts_path.exists():
            with open(refined_prompts_path, 'r') as f:
                refined_prompts = json.load(f)
        else:
            refined_prompts = {}
        
        # Update with new refined prompt
        frame_key = str(request.frame)
        refined_prompts[frame_key] = {
            "refined_prompt": request.prompt,
            "version_filename": request.version_filename,
            "approved": True,
            "timestamp": datetime.now().isoformat(),
            "frame_id": f"{request.frame:04d}"
        }
        
        # Save back to file
        with open(refined_prompts_path, 'w') as f:
            json.dump(refined_prompts, f, indent=2)
        
        logger.info(f"✅ Saved refined prompt for frame {request.frame:04d}")
        
        return SavePromptResponse(success=True)
        
    except Exception as e:
        error_msg = f"Error saving prompt: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return SavePromptResponse(
            success=False,
            error=error_msg
        )

@app.get("/api/prompts/refined")
async def get_refined_prompts():
    """
    Get all refined prompts
    Used by refinery to load saved refinements
    """
    try:
        refined_prompts_path = keyframes_dir.parent / "refined_prompts.json"
        
        if refined_prompts_path.exists():
            with open(refined_prompts_path, 'r') as f:
                refined_prompts = json.load(f)
            return refined_prompts
        else:
            return {}
            
    except Exception as e:
        logger.error(f"Error loading refined prompts: {e}")
        return {}

@app.post("/api/export/shot_list")
async def export_refined_shot_list():
    """
    Export refined shot list with updated prompts
    Creates a new shot list incorporating all approved refinements
    """
    try:
        refined_prompts_path = keyframes_dir.parent / "refined_prompts.json"
        
        if not refined_prompts_path.exists():
            raise HTTPException(status_code=404, detail="No refined prompts found")
        
        with open(refined_prompts_path, 'r') as f:
            refined_prompts = json.load(f)
        
        # Count updated frames
        updated_frames = len([p for p in refined_prompts.values() if p.get('approved', False)])
        
        # Generate export filename
        export_filename = f"refined_shot_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_path = keyframes_dir.parent / export_filename
        
        # Create export data structure
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "updated_frames_count": updated_frames,
            "refined_prompts": refined_prompts,
            "metadata": {
                "pipeline_version": "2.0",
                "scene_id": "intro",
                "export_type": "refined_shot_list"
            }
        }
        
        # Save export file
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"✅ Exported refined shot list: {export_filename}")
        
        return {
            "success": True,
            "filename": export_filename,
            "updated_frames": updated_frames,
            "export_path": str(export_path)
        }
        
    except Exception as e:
        error_msg = f"Export error: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg
        }

@app.get("/api/version/{filename}")
async def get_version_image(filename: str):
    """
    Get specific version of a generated image
    Serves images from the frames directory by filename
    """
    try:
        image_path = keyframes_dir / filename
        
        if image_path.exists():
            return FileResponse(
                image_path,
                media_type="image/png",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            raise HTTPException(status_code=404, detail=f"Version {filename} not found")
            
    except Exception as e:
        logger.error(f"Error serving version {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading version: {str(e)}")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors
    """
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    # Run with uvicorn when executed directly
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )

@app.get("/images/list")
async def list_available_images():
    """List all available keyframe images"""
    if keyframes_dir.exists():
        images = [f.name for f in keyframes_dir.glob("*.png")]
        return {
            "total_images": len(images),
            "images": sorted(images),
            "base_url": "/images/"
        }
    return {"total_images": 0, "images": [], "error": "Images directory not found"}
