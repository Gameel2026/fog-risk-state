@echo off
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
echo === embedded step 1: freeze the deployable model ===
python p4_deploy_model.py > p4_log.txt 2>&1
type p4_log.txt
echo === embedded step 2: Cortex-M4F emulation (about 10-20 min) ===
python mcu\run_fog_mcu.py > mcu_log.txt 2>&1
type mcu_log.txt
echo === done ===
pause
