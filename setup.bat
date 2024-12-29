@echo off
REM setup.bat
ECHO Creating new conda environment gizmo_ar with Python 3.9...
call conda create -n gizmo_ar python=3.9 -y

ECHO Activating environment...
call conda activate gizmo_ar

ECHO Installing pip requirements...
pip install -r requirements.txt

ECHO Setup complete! You can now activate the environment using: conda activate gizmo_ar
pause