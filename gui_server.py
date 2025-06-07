#!/usr/bin/env python3
"""
Simple Flask server for the Keyframe Refinery GUI
"""

from flask import Flask, render_template_string, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import glob
import shutil
from pathlib import Path
from datetime import datetime
import requests
import base64
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
PROJECT_DIR = Path(".")
KEYFRAMES_DIR = PROJECT_DIR / "keyframes"  # Where we'll store current keyframes
VERSIONS_DIR = PROJECT_DIR / "keyframe_versions"  # Where we'll store version history

# Ensure directories exist
KEYFRAMES_DIR.mkdir(exist_ok=True)
VERSIONS_DIR.mkdir(exist_ok=True)

@app.route('/')
def index():
    """Serve the keyframe refinery interface"""
    try:
        with open('keyframe_refinery.html', 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error loading HTML file: {str(e)}", 500

@app.route('/api/frame/<frame_num>')
def get_frame(frame_num):
    """Serve the most recent version of a frame"""
    frame_num_padded = frame_num.zfill(4)
    
    # First check for recent versions in the versions directory
    version_pattern = str(VERSIONS_DIR / f"frame_{frame_num_padded}_*.png")
    version_files = glob.glob(version_pattern)
    
    if version_files:
        # Return the most recent version (sort by filename which includes timestamp)
        latest_version = sorted(version_files)[-1]
        print(f"DEBUG: Serving latest version: {latest_version}")
        return send_file(latest_version)
    
    # Fall back to original in keyframes directory
    keyframe_pattern = str(KEYFRAMES_DIR / f"*{frame_num_padded}*.png")
    keyframe_files = glob.glob(keyframe_pattern)
    
    if keyframe_files:
        print(f"DEBUG: Serving original: {keyframe_files[0]}")
        return send_file(keyframe_files[0])
    
    return jsonify({"error": f"Frame {frame_num} not found"}), 404

@app.route('/api/frame/original/<frame_num>')
def get_original_frame(frame_num):
    """Serve the original frame image, bypassing versions"""
    frame_num_padded = frame_num.zfill(4)
    
    # Only look in keyframes directory for original
    keyframe_pattern = str(KEYFRAMES_DIR / f"*{frame_num_padded}*.png")
    keyframe_files = glob.glob(keyframe_pattern)
    
    if keyframe_files:
        return send_file(keyframe_files[0])
    
    return jsonify({"error": f"Original frame {frame_num} not found"}), 404

@app.route('/api/version/<filename>')
def get_version(filename):
    """Serve a specific version file"""
    version_path = VERSIONS_DIR / filename
    if version_path.exists():
        return send_file(version_path)
    return jsonify({"error": "Version not found"}), 404

@app.route('/api/generate', methods=['POST'])
def generate_frame():
    """Generate a new version of a frame using Stability AI"""
    if not STABILITY_API_KEY:
        return jsonify({"success": False, "error": "Stability API key not configured"})
    
    data = request.json
    frame_num = data.get('frame')
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({"success": False, "error": "No prompt provided"})
    
    # Call Stability AI API
    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": 7,
        "height": 1024,
        "width": 1024,
        "samples": 1,
        "steps": 30
    }
    
    try:
        print(f"DEBUG: Generating for frame {frame_num} with prompt: {prompt[:100]}...")
        response = requests.post(url, headers=headers, json=payload)
        print(f"DEBUG: API response status: {response.status_code}")
        response.raise_for_status()
        
        result = response.json()
        print("DEBUG: API call successful, processing image...")
        
        # Save the generated image
        image_data = base64.b64decode(result["artifacts"][0]["base64"])
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{frame_num:04d}_{timestamp}.png"
        output_path = VERSIONS_DIR / filename
        
        with open(output_path, "wb") as f:
            f.write(image_data)
        
        print(f"DEBUG: Saved new version to {output_path}")
        
        return jsonify({
            "success": True,
            "filename": filename,
            "message": f"Generated new version of frame {frame_num:04d}"
        })
        
    except Exception as e:
        print(f"DEBUG: Error occurred: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/frames/list')
def list_frames():
    """Get list of available frames and their metadata"""
    # Load shot_list.json if it exists
    shot_list_path = PROJECT_DIR / "shot_list.json"
    frame_data = {}
    
    if shot_list_path.exists():
        with open(shot_list_path, 'r') as f:
            shot_list = json.load(f)
            for item in shot_list:
                frame_data[item['global_frame_index']] = {
                    'scene': item['scene'],
                    'description': item['description'],
                    'original_prompt': item['prompt']
                }
    
    return jsonify(frame_data)

@app.route('/api/prompts/save', methods=['POST'])
def save_refined_prompts():
    """Save refined prompts to disk"""
    data = request.json
    frame_num = data.get('frame')
    prompt = data.get('prompt')
    version_filename = data.get('version_filename', None)
    
    # Create refined_prompts.json file
    refined_prompts_file = PROJECT_DIR / "refined_prompts.json"
    
    # Load existing refined prompts or create new structure
    if refined_prompts_file.exists():
        with open(refined_prompts_file, 'r') as f:
            refined_prompts = json.load(f)
    else:
        refined_prompts = {}
    
    # Save the refined prompt
    refined_prompts[str(frame_num)] = {
        'refined_prompt': prompt,
        'version_filename': version_filename,
        'timestamp': datetime.now().isoformat(),
        'approved': True  # Mark as approved version
    }
    
    # Save back to disk
    with open(refined_prompts_file, 'w') as f:
        json.dump(refined_prompts, f, indent=2)
    
    return jsonify({"success": True, "message": f"Refined prompt saved for frame {frame_num}"})

@app.route('/api/prompts/load')
def load_refined_prompts():
    """Load refined prompts from disk"""
    refined_prompts_file = PROJECT_DIR / "refined_prompts.json"
    
    if refined_prompts_file.exists():
        with open(refined_prompts_file, 'r') as f:
            return jsonify(json.load(f))
    else:
        return jsonify({})

@app.route('/api/export/shot_list', methods=['POST'])
def export_updated_shot_list():
    """Export updated shot_list.json with refined prompts"""
    # Load original shot list
    shot_list_path = PROJECT_DIR / "shot_list.json"
    if not shot_list_path.exists():
        return jsonify({"success": False, "error": "Original shot_list.json not found"})
    
    with open(shot_list_path, 'r') as f:
        shot_list = json.load(f)
    
    # Load refined prompts
    refined_prompts_file = PROJECT_DIR / "refined_prompts.json"
    refined_prompts = {}
    if refined_prompts_file.exists():
        with open(refined_prompts_file, 'r') as f:
            refined_prompts = json.load(f)
    
    # Update shot list with refined prompts
    updated_count = 0
    for item in shot_list:
        frame_num = str(item['global_frame_index'])
        if frame_num in refined_prompts:
            item['prompt'] = refined_prompts[frame_num]['refined_prompt']
            item['refined'] = True
            item['refined_timestamp'] = refined_prompts[frame_num]['timestamp']
            updated_count += 1
    
    # Save updated shot list
    updated_filename = f"shot_list_refined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = PROJECT_DIR / updated_filename
    
    with open(output_path, 'w') as f:
        json.dump(shot_list, f, indent=2)
    
    return jsonify({
        "success": True,
        "filename": updated_filename,
        "updated_frames": updated_count,
        "message": f"Exported {updated_count} refined prompts to {updated_filename}"
    })

if __name__ == '__main__':
    import os
    print("Starting Keyframe Refinery Server...")
    print(f"Keyframes directory: {KEYFRAMES_DIR}")
    print(f"Versions directory: {VERSIONS_DIR}")
    
    # Glitch provides PORT environment variable
    port = int(os.environ.get('PORT', 3000))
    
    app.run(debug=False, port=port, host='0.0.0.0')
