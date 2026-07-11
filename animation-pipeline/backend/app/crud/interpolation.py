from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any, Tuple
from models.scenes import Scene, GeneratedFrame, KeyframeSet
from database import get_db
from pathlib import Path
import subprocess
import json
from datetime import datetime


class InterpolationCRUD:
    """CRUD operations for RIFE interpolation jobs and results"""
    
    @staticmethod
    def get_keyframes_for_interpolation(scene_id: str) -> List[Tuple[GeneratedFrame, GeneratedFrame]]:
        """Get consecutive keyframe pairs for interpolation"""
        db = next(get_db())
        
        try:
            # Get latest keyframe set for scene
            latest_set = db.query(KeyframeSet).filter(
                KeyframeSet.scene_id == scene_id
            ).order_by(KeyframeSet.version.desc()).first()
            
            if not latest_set:
                return []
            
            # Get all frames for the set, ordered by beat_index
            frames = db.query(GeneratedFrame).filter(
                GeneratedFrame.keyframe_set_id == latest_set.id
            ).order_by(GeneratedFrame.beat_index).all()
            
            # Create consecutive pairs
            frame_pairs = []
            for i in range(len(frames) - 1):
                frame_pairs.append((frames[i], frames[i + 1]))
            
            return frame_pairs
            
        finally:
            db.close()
    
    @staticmethod
    def calculate_interpolation_plan(scene_id: str, target_fps: int = 24) -> Dict[str, Any]:
        """Calculate how many frames to interpolate between keyframes"""
        db = next(get_db())
        
        try:
            # Get scene info
            scene = db.query(Scene).filter(Scene.id == scene_id).first()
            if not scene:
                return None
            
            # Get keyframe pairs
            frame_pairs = InterpolationCRUD.get_keyframes_for_interpolation(scene_id)
            if not frame_pairs:
                return {"error": "No keyframes available for interpolation"}
            
            interpolation_jobs = []
            total_output_frames = 0
            
            for i, (frame1, frame2) in enumerate(frame_pairs):
                # Calculate time difference between frames
                # This would require beat timestamps - for now use a default
                time_diff = 2.0  # Default 2 seconds between beats
                
                # Calculate frames needed for target FPS
                frames_needed = int(time_diff * target_fps)
                
                # RIFE works best with powers of 2, so find closest exp value
                # 2^exp - 1 = frames_needed, so exp = log2(frames_needed + 1)
                import math
                rife_exp = max(1, int(math.ceil(math.log2(frames_needed + 1))))
                rife_output_frames = (2 ** rife_exp) - 1
                
                job = {
                    "pair_index": i,
                    "frame1_id": frame1.id,
                    "frame2_id": frame2.id,
                    "frame1_filename": frame1.filename,
                    "frame2_filename": frame2.filename,
                    "time_difference": time_diff,
                    "target_frames": frames_needed,
                    "rife_exp": rife_exp,
                    "rife_output_frames": rife_output_frames,
                    "status": "pending"
                }
                
                interpolation_jobs.append(job)
                total_output_frames += rife_output_frames
            
            return {
                "scene_id": scene_id,
                "scene_duration": scene.duration,
                "target_fps": target_fps,
                "keyframe_pairs": len(frame_pairs),
                "interpolation_jobs": interpolation_jobs,
                "total_output_frames": total_output_frames,
                "estimated_video_length": total_output_frames / target_fps
            }
            
        finally:
            db.close()


class RIFEProcessor:
    """Handles RIFE interpolation processing"""
    
    def __init__(self, rife_path: str = "/workspace/RIFE", model_version: str = "4.6"):
        self.rife_path = Path(rife_path)
        self.model_version = model_version
        self.inference_script = self.rife_path / "inference_img.py"
    
    def interpolate_frame_pair(
        self,
        frame1_path: Path,
        frame2_path: Path,
        output_dir: Path,
        exp: int = 2,
        model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Interpolate between two frames using RIFE
        
        Args:
            frame1_path: Path to first keyframe
            frame2_path: Path to second keyframe  
            output_dir: Directory for interpolated frames
            exp: RIFE exp parameter (2^exp - 1 output frames)
            model_path: Custom model path if needed
        """
        
        # Create temporary input directory for RIFE
        temp_input_dir = output_dir / f"rife_input_{frame1_path.stem}_{frame2_path.stem}"
        temp_output_dir = output_dir / f"rife_output_{frame1_path.stem}_{frame2_path.stem}"
        
        temp_input_dir.mkdir(exist_ok=True, parents=True)
        temp_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Copy frames to RIFE input format (0.png, 1.png)
        import shutil
        shutil.copy(frame1_path, temp_input_dir / "0.png")
        shutil.copy(frame2_path, temp_input_dir / "1.png")
        
        # Build RIFE command
        cmd = [
            "python", str(self.inference_script),
            "--exp", str(exp),
            "--img", str(temp_input_dir),
            "--output", str(temp_output_dir)
        ]
        
        if model_path:
            cmd.extend(["--model_path", model_path])
        
        try:
            print(f"Running RIFE interpolation: exp={exp}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5 minute timeout
            )
            
            # Get output frames
            output_frames = sorted(list(temp_output_dir.glob("*.png")))
            
            return {
                "success": True,
                "input_frames": 2,
                "output_frames": len(output_frames),
                "output_paths": [str(p) for p in output_frames],
                "temp_input_dir": str(temp_input_dir),
                "temp_output_dir": str(temp_output_dir),
                "command": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "RIFE processing timeout",
                "timeout": 300
            }
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"RIFE failed with code {e.returncode}",
                "stdout": e.stdout,
                "stderr": e.stderr,
                "command": " ".join(cmd)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "command": " ".join(cmd)
            }
    
    def process_scene_interpolation(
        self,
        scene_id: str,
        keyframes_dir: Path,
        output_dir: Path,
        target_fps: int = 24
    ) -> Dict[str, Any]:
        """Process all interpolation for a scene"""
        
        # Get interpolation plan
        plan = InterpolationCRUD.calculate_interpolation_plan(scene_id, target_fps)
        if not plan or "error" in plan:
            return {"success": False, "error": plan.get("error", "Failed to create plan")}
        
        results = []
        total_processed = 0
        
        for job in plan["interpolation_jobs"]:
            frame1_path = keyframes_dir / job["frame1_filename"]
            frame2_path = keyframes_dir / job["frame2_filename"]
            
            if not frame1_path.exists() or not frame2_path.exists():
                results.append({
                    "pair_index": job["pair_index"],
                    "success": False,
                    "error": "Input frames not found"
                })
                continue
            
            # Run RIFE interpolation
            result = self.interpolate_frame_pair(
                frame1_path,
                frame2_path,
                output_dir,
                exp=job["rife_exp"]
            )
            
            result["pair_index"] = job["pair_index"]
            result["job_info"] = job
            results.append(result)
            
            if result["success"]:
                total_processed += result["output_frames"]
        
        return {
            "success": True,
            "scene_id": scene_id,
            "interpolation_plan": plan,
            "job_results": results,
            "total_frames_processed": total_processed,
            "successful_jobs": sum(1 for r in results if r["success"]),
            "failed_jobs": sum(1 for r in results if not r["success"])
        }


class VideoAssembler:
    """Assembles interpolated frames into final video"""
    
    @staticmethod
    def create_frame_sequence_list(interpolation_results: Dict[str, Any]) -> List[str]:
        """Create ordered list of all frames for video assembly"""
        frame_sequence = []
        
        for result in interpolation_results["job_results"]:
            if result["success"]:
                # Add interpolated frames
                frame_sequence.extend(result["output_paths"])
        
        return frame_sequence
    
    @staticmethod
    def assemble_video_with_ffmpeg(
        frame_sequence: List[str],
        output_path: Path,
        fps: int = 24,
        audio_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Use FFmpeg to create video from frame sequence"""
        
        # Create temporary file list for FFmpeg
        frame_list_file = output_path.parent / f"{output_path.stem}_frames.txt"
        
        with open(frame_list_file, 'w') as f:
            for frame_path in frame_sequence:
                f.write(f"file '{frame_path}'\n")
        
        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y",  # Overwrite output
            "-f", "concat",
            "-safe", "0",
            "-i", str(frame_list_file),
            "-framerate", str(fps),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p"
        ]
        
        if audio_path and audio_path.exists():
            cmd.extend(["-i", str(audio_path), "-c:a", "aac", "-shortest"])
        
        cmd.append(str(output_path))
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            return {
                "success": True,
                "output_path": str(output_path),
                "frame_count": len(frame_sequence),
                "fps": fps,
                "command": " ".join(cmd),
                "stdout": result.stdout
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"FFmpeg failed with code {e.returncode}",
                "stdout": e.stdout,
                "stderr": e.stderr,
                "command": " ".join(cmd)
            }
        finally:
            # Clean up frame list file
            if frame_list_file.exists():
                frame_list_file.unlink()


# Convenience functions
def get_scene_interpolation_status(scene_id: str) -> Dict[str, Any]:
    """Get complete interpolation status for a scene"""
    plan = InterpolationCRUD.calculate_interpolation_plan(scene_id)
    return {
        "interpolation_plan": plan,
        "ready_for_interpolation": plan is not None and "error" not in plan
    }


def create_test_interpolation_job(scene_id: str, rife_path: str = "/workspace/RIFE") -> Dict[str, Any]:
    """Create a test interpolation job for a scene"""
    processor = RIFEProcessor(rife_path)
    
    # This would be called from the runpod environment
    return {
        "message": "Test interpolation job created",
        "rife_processor": "initialized",
        "scene_id": scene_id
    }
