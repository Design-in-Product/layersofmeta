"""
CRUD operations for Scene management in Animation Pipeline 2.0
Handles database operations for scenes, beats, keyframes, and generated frames
"""

from sqlalchemy.orm import Session
from sqlalchemy import desc, asc
from typing import List, Optional, Dict, Any
from datetime import datetime

from models.scenes import Scene, Beat, KeyframeSet, GeneratedFrame
from database import get_db_session


class SceneCRUD:
    """CRUD operations for Scene management"""
    
    @staticmethod
    def create_scene(
        scene_id: str,
        name: str,
        start_time: float,
        duration: float,
        narrative_prompt: str = None,
        style_config: Dict = None
    ) -> Scene:
        """Create a new scene"""
        with get_db_session() as db:
            # Check if scene already exists
            existing = db.query(Scene).filter(Scene.id == scene_id).first()
            if existing:
                raise ValueError(f"Scene with id '{scene_id}' already exists")
            
            # Create new scene
            scene = Scene(
                id=scene_id,
                name=name,
                start_time=start_time,
                duration=duration,
                narrative_prompt=narrative_prompt,
                style_config=style_config or {}
            )
            
            db.add(scene)
            db.commit()
            db.refresh(scene)
            # Make the object independent of the session
            db.expunge(scene)
            return scene
    
    @staticmethod
    def get_scene(scene_id: str) -> Optional[Scene]:
        """Get a scene by ID"""
        with get_db_session() as db:
            scene = db.query(Scene).filter(Scene.id == scene_id).first()
            if scene:
                db.expunge(scene)
            return scene
    
    @staticmethod
    def get_all_scenes() -> List[Scene]:
        """Get all scenes ordered by start_time"""
        with get_db_session() as db:
            scenes = db.query(Scene).order_by(asc(Scene.start_time)).all()
            for scene in scenes:
                db.expunge(scene)
            return scenes
    
    @staticmethod
    def update_scene(
        scene_id: str,
        name: str = None,
        start_time: float = None,
        duration: float = None,
        narrative_prompt: str = None,
        style_config: Dict = None
    ) -> Optional[Scene]:
        """Update an existing scene"""
        with get_db_session() as db:
            scene = db.query(Scene).filter(Scene.id == scene_id).first()
            if not scene:
                return None
            
            # Update fields if provided
            if name is not None:
                scene.name = name
            if start_time is not None:
                scene.start_time = start_time
            if duration is not None:
                scene.duration = duration
            if narrative_prompt is not None:
                scene.narrative_prompt = narrative_prompt
            if style_config is not None:
                scene.style_config = style_config
            
            scene.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(scene)
            return scene
    
    @staticmethod
    def delete_scene(scene_id: str) -> bool:
        """Delete a scene and all its related data"""
        with get_db_session() as db:
            scene = db.query(Scene).filter(Scene.id == scene_id).first()
            if not scene:
                return False
            
            db.delete(scene)
            db.commit()
            return True
    
    @staticmethod
    def add_beat_to_scene(
        scene_id: str,
        timestamp: float,
        description: str,
        characters: List[str] = None,
        beat_order: int = None
    ) -> Optional[Beat]:
        """Add a beat to a scene"""
        with get_db_session() as db:
            scene = db.query(Scene).filter(Scene.id == scene_id).first()
            if not scene:
                return None
            
            # Auto-assign beat order if not provided
            if beat_order is None:
                max_order = db.query(Beat).filter(Beat.scene_id == scene_id).count()
                beat_order = max_order + 1
            
            beat = Beat(
                scene_id=scene_id,
                timestamp=timestamp,
                description=description,
                characters=characters or [],
                beat_order=beat_order
            )
            
            db.add(beat)
            db.commit()
            db.refresh(beat)
            return beat
    
    @staticmethod
    def get_scene_beats(scene_id: str) -> List[Beat]:
        """Get all beats for a scene, ordered by beat_order"""
        with get_db_session() as db:
            return db.query(Beat).filter(Beat.scene_id == scene_id).order_by(asc(Beat.beat_order)).all()
    
    @staticmethod
    def create_keyframe_set(
        scene_id: str,
        version: int = None,
        generation_params: Dict = None
    ) -> Optional[KeyframeSet]:
        """Create a new keyframe set for a scene"""
        with get_db_session() as db:
            scene = db.query(Scene).filter(Scene.id == scene_id).first()
            if not scene:
                return None
            
            # Auto-assign version if not provided
            if version is None:
                max_version = db.query(KeyframeSet).filter(KeyframeSet.scene_id == scene_id).count()
                version = max_version + 1
            
            keyframe_set = KeyframeSet(
                scene_id=scene_id,
                version=version,
                generation_params=generation_params or {}
            )
            
            db.add(keyframe_set)
            db.commit()
            db.refresh(keyframe_set)
            return keyframe_set
    
    @staticmethod
    def get_scene_keyframe_sets(scene_id: str) -> List[KeyframeSet]:
        """Get all keyframe sets for a scene, newest first"""
        with get_db_session() as db:
            return db.query(KeyframeSet).filter(
                KeyframeSet.scene_id == scene_id
            ).order_by(desc(KeyframeSet.version)).all()
    
    @staticmethod
    def add_generated_frame(
        keyframe_set_id: int,
        filename: str,
        beat_index: int = None,
        prompt_used: str = None,
        seed: int = None,
        generation_metadata: Dict = None
    ) -> GeneratedFrame:
        """Add a generated frame to a keyframe set"""
        with get_db_session() as db:
            frame = GeneratedFrame(
                keyframe_set_id=keyframe_set_id,
                filename=filename,
                beat_index=beat_index,
                prompt_used=prompt_used,
                seed=seed,
                generation_metadata=generation_metadata or {}
            )
            
            db.add(frame)
            db.commit()
            db.refresh(frame)
            return frame


# Convenience functions for common operations
def create_scene_with_validation(
    scene_id: str,
    name: str,
    start_time: float,
    duration: float,
    narrative_prompt: str = None,
    style_config: Dict = None
) -> Scene:
    """
    Create a scene with validation
    Validates that scene_id is unique and times don't overlap
    """
    # Validate scene_id format
    if not scene_id or len(scene_id) < 3:
        raise ValueError("Scene ID must be at least 3 characters long")
    
    # Validate timing
    if start_time < 0:
        raise ValueError("Start time must be non-negative")
    if duration <= 0:
        raise ValueError("Duration must be positive")
    
    # Check for overlaps with existing scenes
    existing_scenes = SceneCRUD.get_all_scenes()
    end_time = start_time + duration
    
    for existing in existing_scenes:
        existing_end = existing.start_time + existing.duration
        
        # Check if times overlap
        if (start_time < existing_end and end_time > existing.start_time):
            raise ValueError(
                f"Scene timing overlaps with existing scene '{existing.id}' "
                f"({existing.start_time}-{existing_end}s)"
            )
    
    return SceneCRUD.create_scene(
        scene_id=scene_id,
        name=name,
        start_time=start_time,
        duration=duration,
        narrative_prompt=narrative_prompt,
        style_config=style_config
    )


def get_project_timeline() -> Dict[str, Any]:
    """
    Get complete project timeline with all scenes and their metadata
    Returns a summary suitable for timeline visualization
    """
    scenes = SceneCRUD.get_all_scenes()
    
    timeline = {
        "total_scenes": len(scenes),
        "total_duration": max([s.start_time + s.duration for s in scenes]) if scenes else 0,
        "scenes": []
    }
    
    for scene in scenes:
        beats = SceneCRUD.get_scene_beats(scene.id)
        keyframe_sets = SceneCRUD.get_scene_keyframe_sets(scene.id)
        
        scene_data = scene.to_dict()
        scene_data.update({
            "beats": [beat.to_dict() for beat in beats],
            "keyframe_sets": len(keyframe_sets),
            "latest_keyframe_set": keyframe_sets[0].to_dict() if keyframe_sets else None
        })
        
        timeline["scenes"].append(scene_data)
    
    return timeline
