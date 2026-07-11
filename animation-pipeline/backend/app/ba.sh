# Test import first
python -c "from api.endpoints.interpolation import router; print('✅ Ready to add router!')"

# After adding to main.py, restart and test
python main.py
