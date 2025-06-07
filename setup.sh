# setup.sh
#!/bin/bash
echo "Setting up Saint Lucifer Music Video Pipeline..."

# Create virtual environment
python3 -m venv saint_lucifer_env
source saint_lucifer_env/bin/activate

# Install Python requirements
pip install -r requirements.txt

# Check for FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg not found. Please install FFmpeg:"
    echo "  macOS: brew install ffmpeg"
    echo "  Ubuntu: apt install ffmpeg"
    echo "  Windows: Download from https://ffmpeg.org/"
    exit 1
fi

# Create .env template
cat << EOF > .env.template
# API Keys - Copy this to .env and fill in your keys
STABILITY_API_KEY=your_stability_ai_key_here
MIDJOURNEY_API_KEY=your_midjourney_key_here

# Optional: Alternative APIs
REPLICATE_API_TOKEN=your_replicate_token_here
OPENAI_API_KEY=your_openai_key_here
EOF

echo "Setup complete!"
echo "1. Copy .env.template to .env and add your API keys"
echo "2. Run: python pipeline.py --audio path/to/saint_lucifer.mp3"

# config.yaml - Alternative configuration format
cat << 'EOF' > config.yaml
project:
  name: "saint_lucifer"
  song_duration: 199  # 3:19 in seconds
  fps: 24
  resolution: [1920, 1080]

api_preferences:
  primary: "stability"  # stability, midjourney, replicate
  fallback: "replicate"
  rate_limit_delay: 1  # seconds between requests
  batch_size: 10

style_settings:
  base_quality: "high quality, cinematic composition, dramatic lighting"
  negative_prompt: "blurry, low quality, distorted, ugly, bad anatomy"
  
scenes:
  intro:
    style_weight: 1.2
    cfg_scale: 7
    steps: 30
  verse1:
    style_weight: 1.0
    cfg_scale: 8
    steps: 35
  chorus:
    style_weight: 1.5
    cfg_scale: 9
    steps: 40

output:
  video_codec: "libx264"
  audio_codec: "aac"
  bitrate: "5M"
  preset: "medium"
EOF

echo "Configuration files created!"
