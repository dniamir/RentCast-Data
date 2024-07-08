@echo off

:: Set the desired Python version
SET "PYTHON_VERSION=3.12.4"

:: Check if the specified Python version is installed
for /f "tokens=* USEBACKQ" %%F in (`py -%PYTHON_VERSION:~0,4% -V 2^>^&1`) do (
    SET var=%%F
)
if not "%var%"=="Python %PYTHON_VERSION%" (
    echo Python %PYTHON_VERSION% is not installed
    echo Installing Python %PYTHON_VERSION%...
    powershell -Command "Invoke-WebRequest https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe -OutFile python-%PYTHON_VERSION%-amd64.exe"
    python-%PYTHON_VERSION%-amd64.exe /quiet InstallAllUsers=1 PrependPath=1
    del python-%PYTHON_VERSION%-amd64.exe
) else (
    echo Python %PYTHON_VERSION% is installed
)

:: Create the virtual environment in the parent directory
echo Creating virtual environment in parent directory...
cd ..
py -%PYTHON_VERSION:~0,4% -m venv env
echo Virtual environment created successfully

:: Activate the virtual environment and ensure distutils is installed
call env\Scripts\activate
python -m pip install --upgrade pip
python -m pip install setuptools
python -m pip install distlib

:: Ensure repository main directory can be imported
echo Configuring import paths...
SET var=%cd%
ECHO %var% > env/Lib/site-packages/localpackages.pth
echo Import paths configured successfully

:: Install required packages
echo Installing required packages from requirements.txt...
pip install -r .\Commands\requirements.txt
echo Required packages installed successfully

echo Setup Complete
pause
cmd /k
