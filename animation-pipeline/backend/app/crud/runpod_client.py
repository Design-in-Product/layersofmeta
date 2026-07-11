"""
Runpod RIFE Client - Handles communication between local backend and Runpod RIFE API
Add this to your backend/app/crud/interpolation.py or create a new file
"""

import requests
import base64
import json
import time
import zipfile
import io
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid
from crud.interpolation import InterpolationCRUD

class RunpodRIFEClient:
    """Client for communicating with RIFE API running on Runpod"""
    
    def __init__(self, runpod_api_url: str):
        self.api_url = runpod_api_url.rstrip('/')
        self.session = requests.Session()
        # Set longer timeout for RIFE processing
        self.session.timeout = (10, 300)  # 10s connect, 5min read
    
    def check_health(self) -> Dict[str, Any]:
        """Check if Runpod RIFE API is healthy"""
        try:
            response = self.session.get(f"{self.api_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def encode_image_to_base64(self, image_path: Path) -> str:
        """Convert image file to base64 string for API upload"""
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')
    
    def interpolate_frame_pair(
        self,
        frame1_path: Path,
        frame2_path: Path,
        exp: int = 6,
        output_dir: Path = None
    ) -> Dict[str, Any]:
        """
        Interpolate between two frames using Runpod RIFE API
        
        Args:
            frame1_path: Path to first keyframe
            frame2_path: Path to second keyframe  
            exp: RIFE exp parameter (6 = 64 frames output)
            output_dir: Local directory to save results
        
        Returns:
            Dict with interpolation results and local file paths
        """
        try:
            # Generate unique job ID
            job_id = f"job_{uuid.uuid4().hex[:8]}_{int(time.time())}"
            
            print(f"🎬 Starting RIFE interpolation job: {job_id}")
            print(f"📁 Frame 1: {frame1_path.name}")
            print(f"📁 Frame 2: {frame2_path.name}")
            print(f"⚙️ RIFE exp: {exp} (will generate ~{2**exp} frames)")
            
            # Encode images to base64
            frame1_b64 = self.encode_image_to_base64(frame1_path)
            frame2_b64 = self.encode_image_to_base64(frame2_path)
            
            # Prepare API payload
            payload = {
                "frame1_data": frame1_b64,
                "frame2_data": frame2_b64,
                "exp": exp,
                "job_id": job_id
            }
            
            print("📤 Uploading frames to Runpod...")
            
            # Call RIFE interpolation API
            response = self.session.post(
                f"{self.api_url}/interpolate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"✅ RIFE processing completed!")
            print(f"📊 Generated {result.get('output_frames', 0)} frames")
            
            # Download results if output directory specified
            # if output_dir:
            #     local_files = self.download_job_results(job_id, output_dir)
            #     result['local_files'] = local_files
            #     result['local_output_dir'] = str(output_dir)
            
            return result
            
        except requests.exceptions.Timeout:
            return {"error": "RIFE processing timed out (>5 minutes)"}
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Interpolation failed: {str(e)}"}
    
    def download_job_results(self, job_id: str, output_dir: Path) -> List[str]:
        """Download interpolated frames from Runpod to local directory"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"📥 Downloading results for job {job_id}...")
            
            # Download ZIP file with results
            response = self.session.get(f"{self.api_url}/download/{job_id}")
            response.raise_for_status()
            
            # Extract ZIP contents to output directory
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                local_files = []
                
                for file_info in zip_file.filelist:
                    if file_info.filename.endswith('.png'):
                        # Extract to output directory
                        zip_file.extract(file_info, output_dir)
                        local_file_path = output_dir / file_info.filename
                        local_files.append(str(local_file_path))
                
                print(f"✅ Downloaded {len(local_files)} interpolated frames")
                return sorted(local_files)
                
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return []
    
    def cleanup_job(self, job_id: str) -> bool:
        """Clean up job files on Runpod to save space"""
        try:
            response = self.session.delete(f"{self.api_url}/cleanup/{job_id}")
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"⚠️ Cleanup failed for job {job_id}: {e}")
            return False

# Integration with your existing InterpolationCRUD
class RunpodInterpolationCRUD(InterpolationCRUD):
    """Extended InterpolationCRUD that uses Runpod RIFE API"""
    
    @classmethod
    def process_scene_interpolation_runpod(
        cls,
        scene_id: str,
        runpod_api_url: str,
        keyframes_dir: Path,
        output_dir: Path,
        target_fps: int = 24
    ) -> Dict[str, Any]:
        """
        Process full scene interpolation using Runpod RIFE API
        This replaces the simulate_interpolation with real processing
        """
        try:
            # Initialize Runpod client
            client = RunpodRIFEClient(runpod_api_url)
            
            # Check API health first
            health = client.check_health()
            if health.get('status') != 'healthy':
                return {"error": f"Runpod RIFE API not healthy: {health}"}
            
            print(f"✅ Runpod RIFE API healthy: {health.get('rife_available', False)}")
            
            # Get interpolation plan
            plan = cls.calculate_interpolation_plan(scene_id, target_fps)
            if not plan or "error" in plan:
                return {"error": "Cannot create interpolation plan"}
            
            print(f"📋 Processing {len(plan['interpolation_jobs'])} frame pairs")
            
            # Process each frame pair
            results = []
            total_frames_generated = 0
            
            for job in plan['interpolation_jobs']:
                frame1_path = keyframes_dir / job['frame1_filename']
                frame2_path = keyframes_dir / job['frame2_filename']
                
                # Create job-specific output directory
                pair_output_dir = output_dir / f"pair_{job['pair_index']:03d}"
                
                # Process the pair
                result = client.interpolate_frame_pair(
                    frame1_path=frame1_path,
                    frame2_path=frame2_path,
                    exp=job['rife_exp'],
                    output_dir=pair_output_dir
                )
                
                if 'error' not in result:
                    total_frames_generated += result.get('output_frames', 0)
                    # Clean up Runpod storage after successful download
                    # client.cleanup_job(result['job_id'])
                
                results.append({
                    "pair_index": job['pair_index'],
                    "frame1_filename": job['frame1_filename'],
                    "frame2_filename": job['frame2_filename'],
                    "result": result,
                    "local_output_dir": str(pair_output_dir) if 'error' not in result else None
                })
            
            return {
                "success": True,
                "scene_id": scene_id,
                "total_pairs_processed": len(results),
                "total_frames_generated": total_frames_generated,
                "results": results,
                "output_dir": str(output_dir)
            }
            
        except Exception as e:
            return {"error": f"Scene interpolation failed: {str(e)}"}

# Example usage function for testing
def test_single_pair_interpolation():
    """Test function to verify Runpod integration works"""
    
    # Configuration
    RUNPOD_API_URL = "https://gj8s0v4skzwery-5000.proxy.runpod.net"
    KEYFRAMES_DIR = Path.home() / "Development" / "layersofmeta" / "frames"
    OUTPUT_DIR = Path.home() / "Development" / "layersofmeta" / "frames" / "test_interpolation"
    
    # Initialize client
    client = RunpodRIFEClient(RUNPOD_API_URL)
    
    # Test with your verse1 keyframes
    frame1 = KEYFRAMES_DIR / "gen_verse1_001_39.png"  # Your first verse1 frame
    frame2 = KEYFRAMES_DIR / "gen_verse1_002_39.png"  # Your second verse1 frame
    
    if frame1.exists() and frame2.exists():
        print("🧪 Testing single pair interpolation...")
        result = client.interpolate_frame_pair(
            frame1_path=frame1,
            frame2_path=frame2,
            exp=4,  # Smaller test: 16 frames instead of 64
            output_dir=OUTPUT_DIR
        )
        
        print("🎯 Test Result:")
        print(json.dumps(result, indent=2))
        return result
    else:
        print(f"❌ Test frames not found:")
        print(f"Frame 1: {frame1}")
        print(f"Frame 2: {frame2}")
        return None
