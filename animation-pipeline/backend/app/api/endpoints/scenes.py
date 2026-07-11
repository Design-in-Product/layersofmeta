"""
FastAPI endpoints for Scene management
RESTful API for creating, reading, updating, and deleting scenes
FIXED: Updated to match actual CRUD operations + Scene Iterator support
"""

from fastapi import APIRouter, HTTPException, status
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import sqlite3
from pathlib import Path
import logging
from crud.scenes import SceneCRUD, create_scene_with_validation, get_project_timeline
from database import get_db_session
from models.scenes import Beat  # You'll need this model

# Create router for scene endpoints
router = APIRouter(prefix="/api/scenes", tags=["scenes"])
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class SceneCreate(BaseModel):
    """Request model for creating a new scene"""
    id: str = Field(..., min_length=1, max_length=50, description="Unique scene identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Human-readable scene name")
    start_time: float = Field(..., ge=0, description="Start time in seconds")
    duration: float = Field(..., gt=0, description="Duration in seconds")
    narrative_prompt: Optional[str] = Field(None, description="Overall scene description")
    style_config: Optional[Dict[str, Any]] = Field(None, description="Style configuration")

class SceneUpdate(BaseModel):
    """Request model for updating a scene"""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    start_time: Optional[float] = Field(None, ge=0)
    duration: Optional[float] = Field(None, gt=0)
    narrative_prompt: Optional[str] = None
    style_config: Optional[Dict[str, Any]] = None

class SceneResponse(BaseModel):
    """Response model for scene data"""
    id: str
    name: str
    start_time: float
    duration: float
    end_time: float
    narrative_prompt: Optional[str]
    style_config: Optional[Dict[str, Any]]
    
    @classmethod
    def from_scene(cls, scene):
        """Create response from Scene object"""
        return cls(
            id=scene.id,
            name=scene.name,
            start_time=scene.start_time,
            duration=scene.duration,
            end_time=scene.start_time + scene.duration,
            narrative_prompt=scene.narrative_prompt,
            style_config=scene.style_config
        )

class SceneListResponse(BaseModel):
    """Response model for scene list"""
    scenes: List[SceneResponse]
    total_count: int

# API Endpoints
@router.post("/", response_model=SceneResponse, status_code=status.HTTP_201_CREATED)
async def create_scene(scene_data: SceneCreate):
    """Create a new scene"""
    try:
        scene = create_scene_with_validation(
            scene_id=scene_data.id,
            name=scene_data.name,
            start_time=scene_data.start_time,
            duration=scene_data.duration,
            narrative_prompt=scene_data.narrative_prompt,
            style_config=scene_data.style_config
        )
        return SceneResponse.from_scene(scene)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create scene: {str(e)}")

@router.get("/", response_model=SceneListResponse)
async def list_scenes():
    """Get list of all scenes"""
    try:
        scenes = SceneCRUD.get_all_scenes()
        
        return SceneListResponse(
            scenes=[SceneResponse.from_scene(scene) for scene in scenes],
            total_count=len(scenes)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list scenes: {str(e)}")

@router.get("/timeline")
async def get_project_timeline_endpoint():
    """Get complete project timeline"""
    try:
        scenes = SceneCRUD.get_all_scenes()
        
        timeline = {
            'total_scenes': len(scenes),
            'total_duration': max([s.start_time + s.duration for s in scenes]) if scenes else 0,
            'scenes': [SceneResponse.from_scene(scene).dict() for scene in scenes]
        }
        
        return timeline
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get timeline: {str(e)}")

@router.get("/{scene_id}", response_model=SceneResponse)
async def get_scene(scene_id: str):
    """Get scene by ID"""
    scene = SceneCRUD.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found")
    
    return SceneResponse.from_scene(scene)

@router.put("/{scene_id}", response_model=SceneResponse)
async def update_scene(scene_id: str, scene_data: SceneUpdate):
    """Update scene by ID"""
    try:
        update_data = {k: v for k, v in scene_data.dict().items() if v is not None}
        
        scene = SceneCRUD.update_scene(scene_id, **update_data)
        if not scene:
            raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found")
        
        return SceneResponse.from_scene(scene)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update scene: {str(e)}")

@router.delete("/{scene_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_scene(scene_id: str):
    """Delete scene by ID"""
    try:
        success = SceneCRUD.delete_scene(scene_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found")
        
        return
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete scene: {str(e)}")

# NEW SCENE ITERATOR ENDPOINTS - Add these

@router.get("/{scene_id}/beats")
async def get_scene_beats_for_iterator(scene_id: str):
    """Get beats for Scene Iterator using PostgreSQL"""
    try:
        logger.info(f"Loading beats for scene: {scene_id}")
        
        with get_db_session() as db:
            # Query beats using SQLAlchemy ORM
            beats_query = db.query(Beat).filter(Beat.scene_id == scene_id).order_by(Beat.beat_order.asc(), Beat.id.asc()).all()
            
            # Convert to the format Scene Iterator expects
            processed_beats = []
            seen_descriptions = set()
            
            for beat in beats_query:
                # Skip duplicate descriptions
                if beat.description in seen_descriptions:
                    continue
                seen_descriptions.add(beat.description)
                
                # Generate prompt from description since there's no prompt field
                base_prompt = f"extreme close-up {beat.description.lower()}, detailed photorealistic woodgrain, stark black background, film noir chiaroscuro lighting"
                
                processed_beat = {
                    'id': beat.id,
                    'scene_id': beat.scene_id,
                    'beat_order': beat.beat_order,
                    'timestamp': beat.timestamp or 0,
                    'description': beat.description or 'No description',
                    'prompt': base_prompt,  # Generated from description
                    'refined_prompt': None,  # Not available in this schema
                    'quality_approved': False,  # Not available in this schema
                    'created_at': None  # Not available in this schema
                }
                
                processed_beats.append(processed_beat)
            
            logger.info(f"✅ Returning {len(processed_beats)} processed beats for {scene_id}")
            return processed_beats
            
    except Exception as e:
        logger.error(f"❌ Error fetching beats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch beats: {str(e)}")
        
@router.get("/{scene_id}/status")
async def get_scene_status_for_iterator(scene_id: str):
    """Get scene status for Scene Iterator using PostgreSQL"""
    try:
        with get_db_session() as db:
            # Get scene statistics using actual Beat fields
            total_beats = db.query(Beat).filter(Beat.scene_id == scene_id).count()
            
            # Since quality_approved doesn't exist, assume no beats are locked yet
            locked_beats = 0
            
            # Get beats for locks (using logical IDs)
            beats_query = db.query(Beat).filter(Beat.scene_id == scene_id).order_by(Beat.beat_order.asc(), Beat.id.asc()).all()
            
            locks = {}
            for i, beat in enumerate(beats_query):
                locks[i] = {
                    'locked': False,  # Default to unlocked
                    'locked_at': None,
                    'database_id': beat.id
                }
            
            # Calculate completion rate based on existing images
            frames_dir = Path.home() / "Development" / "layersofmeta" / "frames"
            existing_images = len([f for f in frames_dir.glob("*.png")]) if frames_dir.exists() else 0
            completion_rate = min(100, int((existing_images / total_beats) * 100)) if total_beats > 0 else 0
            
            return {
                "scene_id": scene_id,
                "total_beats": total_beats,
                "existing_images": existing_images,
                "locked_beats": locked_beats,
                "completion_rate": completion_rate,
                "ready_for_rife": False,  # Since nothing is locked yet
                "locks": locks
            }
            
    except Exception as e:
        logger.error(f"❌ Status error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get scene status: {str(e)}")
        
# Health check endpoint
@router.get("/health", include_in_schema=False)
async def health_check():
    """Simple health check for the scenes API"""
    return {"status": "healthy", "service": "scenes_api"}
