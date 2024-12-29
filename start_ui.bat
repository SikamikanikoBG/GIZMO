@echo off
REM setup.bat

ECHO Activating environment...
call conda activate gizmo_ar

ECHO Running the UI...
python gizmo_ui.py