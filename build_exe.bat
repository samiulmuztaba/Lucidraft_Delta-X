@echo off
echo Building Lucidraft Executable...

REM Install requirements
pip install -r requirements.txt

REM Create the executable with PyInstaller
pyinstaller --onefile --console --name "Lucidraft" --add-data "demo_flight1.mp4;." --add-data "demo_flight2.mp4;." --add-data "demo_aircraft.jpg;." lucidraft.py

echo.
echo Build complete! Check the 'dist' folder for Lucidraft.exe
pause