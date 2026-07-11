i# From your backend directory, test the exact path calculation
cd ~/Development/layersofmeta/animation-pipeline/backend/app
python3 -c "from pathlib import Path; print(Path(__file__).parent.parent.parent); print((Path(__file__).parent.parent.parent / 'keyframe_refinery.html').exists())"
