from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from crud.beats import BeatCRUD

router = APIRouter()


# Pydantic models for request/response
class BeatCreate(BaseModel):
    scene_id: str
    timestamp: float
    description: str
    characters: List[str] = []
    beat_order: Optional[int] = None


class BeatUpdate(BaseModel):
    timestamp: Optional[float] = None
    description: Optional[str] = None
    characters: Optional[List[str]] = None
    beat_order: Optional[int] = None


class BeatResponse(BaseModel):
    id: int
    scene_id: str
    timestamp: float
    description: str
    characters: List[str]
    beat_order: int

    class Config:
        from_attributes = True


class BeatReorder(BaseModel):
    scene_id: str
    new_order: List[int]  # List of beat IDs in new order


# CRUD Endpoints
@router.post("/beats/", response_model=BeatResponse)
async def create_beat(beat_data: BeatCreate):
    """Create a new beat within a scene"""
    try:
        beat = BeatCRUD.create_beat(
            scene_id=beat_data.scene_id,
            timestamp=beat_data.timestamp,
            description=beat_data.description,
            characters=beat_data.characters,
            beat_order=beat_data.beat_order
        )
        return beat
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/beats/{beat_id}", response_model=BeatResponse)
async def get_beat(beat_id: int):
    """Get a specific beat by ID"""
    beat = BeatCRUD.get_beat_by_id(beat_id)
    if not beat:
        raise HTTPException(status_code=404, detail="Beat not found")
    return beat


@router.put("/beats/{beat_id}", response_model=BeatResponse)
async def update_beat(beat_id: int, beat_update: BeatUpdate):
    """Update an existing beat"""
    beat = BeatCRUD.update_beat(
        beat_id=beat_id,
        timestamp=beat_update.timestamp,
        description=beat_update.description,
        characters=beat_update.characters,
        beat_order=beat_update.beat_order
    )
    if not beat:
        raise HTTPException(status_code=404, detail="Beat not found")
    return beat


@router.delete("/beats/{beat_id}")
async def delete_beat(beat_id: int):
    """Delete a beat"""
    success = BeatCRUD.delete_beat(beat_id)
    if not success:
        raise HTTPException(status_code=404, detail="Beat not found")
    return {"message": "Beat deleted successfully"}


# Scene-specific endpoints
@router.get("/scenes/{scene_id}/beats", response_model=List[BeatResponse])
async def get_scene_beats(scene_id: str):
    """Get all beats for a specific scene"""
    beats = BeatCRUD.get_beats_for_scene(scene_id)
    return beats


@router.get("/scenes/{scene_id}/beats/timeline")
async def get_scene_timeline(scene_id: str):
    """Get complete timeline data for a scene including beats"""
    timeline = BeatCRUD.get_scene_timeline(scene_id)
    if not timeline:
        raise HTTPException(status_code=404, detail="Scene not found")
    return timeline


@router.get("/scenes/{scene_id}/beats/timerange")
async def get_beats_by_timerange(
    scene_id: str,
    start_time: float = Query(..., description="Start time in seconds"),
    end_time: float = Query(..., description="End time in seconds")
):
    """Get beats within a specific time range"""
    beats = BeatCRUD.get_beats_by_timestamp(scene_id, start_time, end_time)
    return beats


@router.put("/scenes/{scene_id}/beats/reorder")
async def reorder_scene_beats(scene_id: str, reorder_data: BeatReorder):
    """Reorder beats within a scene"""
    if reorder_data.scene_id != scene_id:
        raise HTTPException(status_code=400, detail="Scene ID mismatch")
    
    try:
        updated_beats = BeatCRUD.reorder_beats(scene_id, reorder_data.new_order)
        return {
            "message": "Beats reordered successfully",
            "beats": updated_beats
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Bulk operations
@router.post("/scenes/{scene_id}/beats/bulk", response_model=List[BeatResponse])
async def create_multiple_beats(scene_id: str, beats_data: List[BeatCreate]):
    """Create multiple beats for a scene at once"""
    created_beats = []
    
    for beat_data in beats_data:
        # Ensure scene_id matches
        beat_data.scene_id = scene_id
        
        try:
            beat = BeatCRUD.create_beat(
                scene_id=beat_data.scene_id,
                timestamp=beat_data.timestamp,
                description=beat_data.description,
                characters=beat_data.characters,
                beat_order=beat_data.beat_order
            )
            created_beats.append(beat)
        except Exception as e:
            # If one fails, rollback might be needed - for now, continue
            continue
    
    return created_beats


# Analysis endpoints
@router.get("/scenes/{scene_id}/beats/stats")
async def get_beat_statistics(scene_id: str):
    """Get statistics about beats in a scene"""
    beats = BeatCRUD.get_beats_for_scene(scene_id)
    
    if not beats:
        return {
            "total_beats": 0,
            "average_spacing": 0,
            "characters_involved": [],
            "timeline_coverage": 0
        }
    
    # Calculate statistics
    timestamps = [beat.timestamp for beat in beats]
    all_characters = set()
    for beat in beats:
        all_characters.update(beat.characters)
    
    timeline_coverage = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
    average_spacing = timeline_coverage / (len(beats) - 1) if len(beats) > 1 else 0
    
    return {
        "total_beats": len(beats),
        "average_spacing": round(average_spacing, 2),
        "characters_involved": list(all_characters),
        "timeline_coverage": round(timeline_coverage, 2),
        "first_beat": min(timestamps),
        "last_beat": max(timestamps)
    }
