@echo off
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
set LOKY_MAX_CPU_COUNT=4
echo === Amendment 02: single real run ===
python p3_amend02.py > amend02_log.txt 2>&1
type amend02_log.txt
echo === done ===
pause
