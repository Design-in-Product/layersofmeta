from sqlalchemy.orm import Session
from sqlalchemy import and_
from typing import List, Optional, Dict, Any
from models.scenes import Beat, Scene
from database import get_db


class BeatCRUD:
    """CRUD operations for beats within scenes"""
    
    @staticmethod
    def create_beat(
        scene_id: str,
        timestamp: float,
        description: str,
        characters: List[str] = None,
        beat_order: int = None
    ) -> Beat:
        """Create a new beat within a scene"""
        db = next(get_db())
        
        try:
            # Auto-assign beat_order if not provided
            if beat_order is None:
                max_order = db.query(Beat).filter(
                    Beat.scene_id == scene_id
                ).count()
                beat_order = max_order + 1
            
            # Create new beat
            new_beat = Beat(
                scene_id=scene_id,
                timestamp=timestamp,
                description=description,
                characters=characters or [],
                beat_order=beat_order
            )
            
            db.add(new_beat)
            db.commit()
            db.refresh(new_beat)
            
            return new_beat
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    @staticmethod
    def get_beats_for_scene(scene_id: str) -> List[Beat]:
        """Get all beats for a specific scene, ordered by beat_order"""
        db = next(get_db())
        
        try:
            beats = db.query(Beat).filter(
                Beat.scene_id == scene_id
            ).order_by(Beat.beat_order).all()
            
            return beats
            
        finally:
            db.close()
    
    @staticmethod
    def get_beat_by_id(beat_id: int) -> Optional[Beat]:
        """Get a specific beat by ID"""
        db = next(get_db())
        
        try:
            beat = db.query(Beat).filter(Beat.id == beat_id).first()
            return beat
            
        finally:
            db.close()
    
    @staticmethod
    def get_beats_by_timestamp(scene_id: str, start_time: float, end_time: float) -> List[Beat]:
        """Get beats within a specific time range in a scene"""
        db = next(get_db())
        
        try:
            beats = db.query(Beat).filter(
                and_(
                    Beat.scene_id == scene_id,
                    Beat.timestamp >= start_time,
                    Beat.timestamp <= end_time
                )
            ).order_by(Beat.timestamp).all()
            
            return beats
            
        finally:
            db.close()
    
    @staticmethod
    def update_beat(
        beat_id: int,
        timestamp: float = None,
        description: str = None,
        characters: List[str] = None,
        beat_order: int = None
    ) -> Optional[Beat]:
        """Update an existing beat"""
        db = next(get_db())
        
        try:
            beat = db.query(Beat).filter(Beat.id == beat_id).first()
            
            if not beat:
                return None
            
            # Update provided fields
            if timestamp is not None:
                beat.timestamp = timestamp
            if description is not None:
                beat.description = description
            if characters is not None:
                beat.characters = characters
            if beat_order is not None:
                beat.beat_order = beat_order
            
            db.commit()
            db.refresh(beat)
            
            return beat
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    @staticmethod
    def delete_beat(beat_id: int) -> bool:
        """Delete a beat by ID"""
        db = next(get_db())
        
        try:
            beat = db.query(Beat).filter(Beat.id == beat_id).first()
            
            if not beat:
                return False
            
            db.delete(beat)
            db.commit()
            
            return True
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    @staticmethod
    def reorder_beats(scene_id: str, new_order: List[int]) -> List[Beat]:
        """Reorder beats within a scene"""
        db = next(get_db())
        
        try:
            # Update beat_order for each beat according to new_order list
            for index, beat_id in enumerate(new_order):
                beat = db.query(Beat).filter(Beat.id == beat_id).first()
                if beat and beat.scene_id == scene_id:
                    beat.beat_order = index + 1
            
            db.commit()
            
            # Return updated beats
            updated_beats = db.query(Beat).filter(
                Beat.scene_id == scene_id
            ).order_by(Beat.beat_order).all()
            
            return updated_beats
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    @staticmethod
    def get_scene_timeline(scene_id: str) -> Dict[str, Any]:
        """Get complete timeline data for a scene including beats"""
        db = next(get_db())
        
        try:
            # Get scene info
            scene = db.query(Scene).filter(Scene.id == scene_id).first()
            if not scene:
                return None
            
            # Get all beats for scene
            beats = db.query(Beat).filter(
                Beat.scene_id == scene_id
            ).order_by(Beat.beat_order).all()
            
            # Format timeline data
            timeline = {
                "scene": {
                    "id": scene.id,
                    "name": scene.name,
                    "start_time": scene.start_time,
                    "duration": scene.duration,
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
                ],
                "total_beats": len(beats)
            }
            
            return timeline
            
        finally:
            db.close()


# Convenience functions for common operations
def add_beat_to_scene(scene_id: str, timestamp: float, description: str,
                     characters: List[str] = None) -> Beat:
    """Quick function to add a beat to a scene"""
    return BeatCRUD.create_beat(scene_id, timestamp, description, characters)


def get_scene_beats(scene_id: str) -> List[Beat]:
    """Quick function to get all beats for a scene"""
    return BeatCRUD.get_beats_for_scene(scene_id)
