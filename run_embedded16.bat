@echo off
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
echo === embedded 16 Hz step 1: deployable 16-Hz model ===
python p4b_deploy_model_fs16.py > p4b_log.txt 2>&1
type p4b_log.txt
echo === embedded 16 Hz step 2: Cortex-M4F emulation ===
python mcu16\run_fog_mcu16.py > mcu16_log.txt 2>&1
type mcu16_log.txt
echo === done ===
pause
