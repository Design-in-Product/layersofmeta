"""
Database models for Animation Pipeline 2.0
SQLAlchemy models for scenes, beats, keyframes, and generated frames
FIXED: Renamed 'metadata' column to 'generation_metadata' to avoid SQLAlchemy conflict
"""

from sqlalchemy import Column, String, Float, Text, Integer, JSON, DateTime, ForeignKey, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, List, Optional

Base = declarative_base()

class Scene(Base):
    """
    Scene model - represents a single scene in the music video
    Each scene has timing, narrative prompt, and style configuration
    """
    __tablename__ = "scenes"
    
    id = Column(String(50), primary_key=True)  # e.g., "verse1_lucifer_entrance"
    name = Column(String(200), nullable=False)  # Human-readable name
    start_time = Column(Float, nullable=False)  # In seconds
    duration = Column(Float, nullable=False)    # In seconds
    narrative_prompt = Column(Text)             # Overall scene description
    style_config = Column(JSON)                 # Style descriptors and config
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    beats = relationship("Beat", back_populates="scene", cascade="all, delete-orphan")
    keyframe_sets = relationship("KeyframeSet", back_populates="scene", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Scene(id='{self.id}', name='{self.name}', start_time={self.start_time})>"
    
    @property
    def end_time(self) -> float:
        """Calculate end time from start_time + duration"""
        return self.start_time + self.duration
    
    @property
    def fps(self) -> int:
        """Default FPS for scene (can be overridden in style_config)"""
        if self.style_config and 'fps' in self.style_config:
            return self.style_config['fps']
        return 24
    
    def to_dict(self) -> Dict:
        """Convert scene to dictionary representation"""
        return {
            'id': self.id,
            'name': self.name,
            'start_time': self.start_time,
            'duration': self.duration,
            'end_time': self.end_time,
            'narrative_prompt': self.narrative_prompt,
            'style_config': self.style_config,
            'fps': self.fps,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'beats_count': len(self.beats) if self.beats else 0,
            'keyframe_sets_count': len(self.keyframe_sets) if self.keyframe_sets else 0
        }

class Beat(Base):
    """
    Beat model - represents a timed story moment within a scene
    Each beat has a timestamp relative to scene start and describes what happens
    """
    __tablename__ = "beats"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    scene_id = Column(String(50), ForeignKey('scenes.id'), nullable=False)
    timestamp = Column(Float, nullable=False)   # Relative to scene start
    description = Column(Text)                  # What happens at this moment
    characters = Column(JSON)                   # List of characters in frame
    beat_order = Column(Integer, nullable=False) # Order within scene
    
    # Relationships
    scene = relationship("Scene", back_populates="beats")
    
    def __repr__(self):
        return f"<Beat(id={self.id}, scene_id='{self.scene_id}', timestamp={self.timestamp})>"
    
    @property
    def absolute_timestamp(self) -> float:
        """Calculate absolute timestamp (scene start + beat timestamp)"""
        if self.scene:
            return self.scene.start_time + self.timestamp
        return self.timestamp
    
    def to_dict(self) -> Dict:
        """Convert beat to dictionary representation"""
        return {
            'id': self.id,
            'scene_id': self.scene_id,
            'timestamp': self.timestamp,
            'absolute_timestamp': self.absolute_timestamp,
            'description': self.description,
            'characters': self.characters,
            'beat_order': self.beat_order
        }

class KeyframeSet(Base):
    """
    KeyframeSet model - represents a versioned set of generated keyframes for a scene
    Supports multiple generations/versions per scene for comparison and rollback
    """
    __tablename__ = "keyframe_sets"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    scene_id = Column(String(50), ForeignKey('scenes.id'), nullable=False)
    version = Column(Integer, nullable=False)   # Version number for this scene
    generation_params = Column(JSON)            # Parameters used for generation
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    scene = relationship("Scene", back_populates="keyframe_sets")
    frames = relationship("GeneratedFrame", back_populates="keyframe_set", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<KeyframeSet(id={self.id}, scene_id='{self.scene_id}', version={self.version})>"
    
    def to_dict(self) -> Dict:
        """Convert keyframe set to dictionary representation"""
        return {
            'id': self.id,
            'scene_id': self.scene_id,
            'version': self.version,
            'generation_params': self.generation_params,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'frames_count': len(self.frames) if self.frames else 0,
            'frames': [frame.to_dict() for frame in self.frames] if self.frames else []
        }

class GeneratedFrame(Base):
    """
    GeneratedFrame model - represents a single generated frame within a keyframe set
    Stores metadata about how the frame was generated and links to the file
    """
    __tablename__ = "generated_frames"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    keyframe_set_id = Column(Integer, ForeignKey('keyframe_sets.id'), nullable=False)
    filename = Column(String(255), nullable=False)  # Path to generated image file
    beat_index = Column(Integer)                    # Which beat this frame represents
    prompt_used = Column(Text)                      # Full prompt used for generation
    seed = Column(BigInteger)                       # Seed for reproducibility
    generation_metadata = Column(JSON)              # FIXED: Renamed from 'metadata' to 'generation_metadata'
    
    # Relationships
    keyframe_set = relationship("KeyframeSet", back_populates="frames")
    
    def __repr__(self):
        return f"<GeneratedFrame(id={self.id}, filename='{self.filename}', beat_index={self.beat_index})>"
    
    @property
    def scene_id(self) -> Optional[str]:
        """Get scene ID through keyframe set relationship"""
        if self.keyframe_set:
            return self.keyframe_set.scene_id
        return None
    
    def to_dict(self) -> Dict:
        """Convert generated frame to dictionary representation"""
        return {
            'id': self.id,
            'keyframe_set_id': self.keyframe_set_id,
            'scene_id': self.scene_id,
            'filename': self.filename,
            'beat_index': self.beat_index,
            'prompt_used': self.prompt_used,
            'seed': self.seed,
            'generation_metadata': self.generation_metadata  # FIXED: Updated field name
        }

# Utility functions for model operations
def create_tables(engine):
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def drop_tables(engine):
    """Drop all tables in the database"""
    Base.metadata.drop_all(bind=engine)