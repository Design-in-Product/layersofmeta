# Delete the duplicate beats first to clean up
python -c "
from crud.beats import BeatCRUD
BeatCRUD.delete_beat(2)
BeatCRUD.delete_beat(3)
print('Cleaned up duplicate beats')
"
