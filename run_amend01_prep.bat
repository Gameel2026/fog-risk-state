@echo off
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
set LOKY_MAX_CPU_COUNT=4
echo === Amendment 01 prep: FoG distances (label-free) ===
python p1b_fog_distance.py > p1b_log.txt 2>&1
echo === Amendment 01 prep: dry run with shuffled labels ===
python p3_amend01.py --dryrun > amend01_dry_log.txt 2>&1
type amend01_dry_log.txt
echo === done ===
pause
