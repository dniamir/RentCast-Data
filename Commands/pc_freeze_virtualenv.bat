@echo off

:: Change directory to the parent directory where the virtual environment is located
cd ..

:: Activate the virtual environment
call env\Scripts\activate

:: Write all currently installed packages to requirements.txt
echo Writing current packages to requirements.txt...
pip freeze > .\Commands\requirements.txt
echo Current packages written to requirements.txt successfully

pause
cmd /k
