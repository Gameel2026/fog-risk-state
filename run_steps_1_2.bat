@echo off
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
echo === step 1: build freeze-free windows ===
python p1_build_windows.py > step1_log.txt 2>&1
echo === step 2: freeze the gait gate ===
python p2_freeze_gate.py > step2_log.txt 2>&1
type step2_log.txt
echo === done ===
pause
