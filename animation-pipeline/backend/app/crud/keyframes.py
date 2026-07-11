from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from models.scenes import KeyframeSet, GeneratedFrame, Scene, Beat
from database import get_db
import json
from datetime import datetime


class KeyframeCRUD:
    """CRUD operations for keyframe sets and generated frames"""
    
    @staticmethod
    def create_keyframe_set(
        scene_id: str,
        generation_params: Dict[str, Any],
        version: int = None
    ) -> KeyframeSet:
        """Create a new keyframe set for a scene"""
        db = next(get_db())
        
        try:
            # Auto-assign version if not provided
            if version is None:
                max_version = db.query(KeyframeSet).filter(
                    KeyframeSet.scene_id == scene_id
                ).count()
                version = max_version + 1
            
            new_keyframe_set = KeyframeSet(
                scene_id=scene_id,
                version=version,
                generation_params=generation_params,
                created_at=datetime.utcnow()
            )
            
            db.add(new_keyframe_set)
            db.commit()
            db.refresh(new_keyframe_set)
            
            return new_keyframe_set
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    @staticmethod
    def add_generated_frame(
        keyframe_set_id: int,
        filename: str,
        beat_index: int,
        prompt_used: str,
        seed: int = None,
        metadata: Dict[str, Any] = None
    ) -> GeneratedFrame:
        """Add a generated frame to a keyframe set"""
        db = next(get_db())
        
        try:
            new_frame = GeneratedFrame(
                keyframe_set_id=keyframe_set_id,
                filename=filename,
                beat_index=beat_index,
                prompt_used=prompt_used,
                seed=seed,
                metadata=metadata or {}
            )
            
            db.add(new_frame)
            db.commit()
            db.refresh(new_frame)
            
            return new_frame
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    @staticmethod
    def get_keyframe_sets_for_scene(scene_id: str) -> List[KeyframeSet]:
        """Get all keyframe sets for a scene"""
        db = next(get_db())
        
        try:
            keyframe_sets = db.query(KeyframeSet).filter(
                KeyframeSet.scene_id == scene_id
            ).order_by(KeyframeSet.version.desc()).all()
            
            return keyframe_sets
            
        finally:
            db.close()
    
    @staticmethod
    def get_keyframe_set_by_id(keyframe_set_id: int) -> Optional[KeyframeSet]:
        """Get a specific keyframe set by ID"""
        db = next(get_db())
        
        try:
            keyframe_set = db.query(KeyframeSet).filter(
                KeyframeSet.id == keyframe_set_id
            ).first()
            return keyframe_set
            
        finally:
            db.close()
    
    @staticmethod
    def get_frames_for_keyframe_set(keyframe_set_id: int) -> List[GeneratedFrame]:
        """Get all generated frames for a keyframe set"""
        db = next(get_db())
        
        try:
            frames = db.query(GeneratedFrame).filter(
                GeneratedFrame.keyframe_set_id == keyframe_set_id
            ).order_by(GeneratedFrame.beat_index).all()
            
            return frames
            
        finally:
            db.close()
    
    @staticmethod
    def get_latest_keyframe_set(scene_id: str) -> Optional[KeyframeSet]:
        """Get the most recent keyframe set for a scene"""
        db = next(get_db())
        
        try:
            latest_set = db.query(KeyframeSet).filter(
                KeyframeSet.scene_id == scene_id
            ).order_by(KeyframeSet.version.desc()).first()
            
            return latest_set
            
        finally:
            db.close()
    
    @staticmethod
    def get_scene_generation_status(scene_id: str) -> Dict[str, Any]:
        """Get complete generation status for a scene"""
        db = next(get_db())
        
        try:
            # Get scene info
            scene = db.query(Scene).filter(Scene.id == scene_id).first()
            if not scene:
                return None
            
            # Get beats for the scene
            beats = db.query(Beat).filter(
                Beat.scene_id == scene_id
            ).order_by(Beat.beat_order).all()
            
            # Get latest keyframe set
            latest_keyframe_set = db.query(KeyframeSet).filter(
                KeyframeSet.scene_id == scene_id
            ).order_by(KeyframeSet.version.desc()).first()
            
            # Get generated frames if keyframe set exists
            generated_frames = []
            if latest_keyframe_set:
                generated_frames = db.query(GeneratedFrame).filter(
                    GeneratedFrame.keyframe_set_id == latest_keyframe_set.id
                ).order_by(GeneratedFrame.beat_index).all()
            
            # Calculate generation progress
            total_beats = len(beats)
            generated_count = len(generated_frames)
            progress_percentage = (generated_count / total_beats * 100) if total_beats > 0 else 0
            
            return {
                "scene": {
                    "id": scene.id,
                    "name": scene.name,
                    "start_time": scene.start_time,
                    "duration": scene.duration
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
                "keyframe_set": {
                    "id": latest_keyframe_set.id,
                    "version": latest_keyframe_set.version,
                    "generation_params": latest_keyframe_set.generation_params,
                    "created_at": latest_keyframe_set.created_at.isoformat()
                } if latest_keyframe_set else None,
                "generated_frames": [
                    {
                        "id": frame.id,
                        "filename": frame.filename,
                        "beat_index": frame.beat_index,
                        "prompt_used": frame.prompt_used,
                        "seed": frame.seed,
                        "metadata": frame.metadata
                    }
                    for frame in generated_frames
                ],
                "progress": {
                    "total_beats": total_beats,
                    "generated_frames": generated_count,
                    "percentage": round(progress_percentage, 2),
                    "status": "complete" if generated_count == total_beats else "in_progress" if generated_count > 0 else "not_started"
                }
            }
            
        finally:
            db.close()
    
    @staticmethod
    def delete_keyframe_set(keyframe_set_id: int) -> bool:
        """Delete a keyframe set and all its generated frames"""
        db = next(get_db())
        
        try:
            # Delete all frames first (foreign key constraint)
            db.query(GeneratedFrame).filter(
                GeneratedFrame.keyframe_set_id == keyframe_set_id
            ).delete()
            
            # Delete the keyframe set
            keyframe_set = db.query(KeyframeSet).filter(
                KeyframeSet.id == keyframe_set_id
            ).first()
            
            if not keyframe_set:
                return False
            
            db.delete(keyframe_set)
            db.commit()
            
            return True
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()


# Convenience functions
def create_keyframe_generation_for_scene(scene_id: str, api_provider: str = "stability") -> KeyframeSet:
    """Create a new keyframe set for scene generation"""
    generation_params = {
        "api_provider": api_provider,
        "model": "stable-diffusion-xl-1024-v1-0" if api_provider == "stability" else "unknown",
        "resolution": "1024x1024",
        "steps": 30,
        "cfg_scale": 7
    }
    
    return KeyframeCRUD.create_keyframe_set(scene_id, generation_params)


def add_frame_to_generation(keyframe_set_id: int, filename: str, beat_index: int, prompt: str) -> GeneratedFrame:
    """Quick function to add a generated frame"""
    return KeyframeCRUD.add_generated_frame(keyframe_set_id, filename, beat_index, prompt)
