@echo off
cd /d "%~dp0"
copy /Y p1c_build_notype_v2.py p1c_build_notype.py
copy /Y p3_amend02_v2.py p3_amend02.py
del p1c_build_notype_v2.py p3_amend02_v2.py
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
set LOKY_MAX_CPU_COUNT=4
echo === Amendment 02 prep: windows of the notype recordings (label-free) ===
python p1c_build_notype.py > p1c_log.txt 2>&1
type p1c_log.txt
echo === Amendment 02 prep: dry run with shuffled labels (counts only) ===
python p3_amend02.py --dryrun > amend02_dry_log.txt 2>&1
type amend02_dry_log.txt
echo === done ===
pause
