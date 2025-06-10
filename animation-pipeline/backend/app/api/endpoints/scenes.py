"""
FastAPI endpoints for Scene management
RESTful API for creating, reading, updating, and deleting scenes
FIXED: Updated to match actual CRUD operations
"""

from fastapi import APIRouter, HTTPException, status
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from crud.scenes import SceneCRUD, create_scene_with_validation, get_project_timeline

# Create router for scene endpoints
router = APIRouter(prefix="/api/scenes", tags=["scenes"])

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
    """
    Create a new scene
    
    Creates a new scene with timing validation to ensure no conflicts.
    """
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
    """
    Get list of all scenes
    
    Returns all scenes ordered by start time.
    """
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
    """
    Get complete project timeline
    
    Returns all scenes with timing information for project overview.
    """
    try:
        # Use our simple timeline that doesn't rely on relationships
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
    """
    Get scene by ID
    
    Returns detailed information about a specific scene.
    """
    scene = SceneCRUD.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found")
    
    return SceneResponse.from_scene(scene)

@router.put("/{scene_id}", response_model=SceneResponse)
async def update_scene(scene_id: str, scene_data: SceneUpdate):
    """
    Update scene by ID
    
    Updates scene fields. Note: timing validation is basic for now.
    """
    try:
        # Prepare update data - only include non-None values
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
    """
    Delete scene by ID
    
    Deletes scene and all associated beats, keyframe sets, and generated frames.
    """
    try:
        success = SceneCRUD.delete_scene(scene_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found")
        
        # 204 No Content - don't return anything
        return
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete scene: {str(e)}")

# Health check endpoint
@router.get("/health", include_in_schema=False)
async def health_check():
    """Simple health check for the scenes API"""
    return {"status": "healthy", "service": "scenes_api"}
