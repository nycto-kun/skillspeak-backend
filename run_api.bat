@echo off
echo Installing Python dependencies...
pip install -r requirements.txt

echo Starting Skillspeak API...
python main.py

pause