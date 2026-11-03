@echo off
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
echo === step 3: medication-state analysis (may take 30-90 min) ===
python p3_analysis.py > step3_log.txt 2>&1
type step3_log.txt
echo === done ===
pause
