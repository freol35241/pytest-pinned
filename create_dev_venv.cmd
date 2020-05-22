@ECHO OFF

ECHO Creating virtual environment

py -3-32 -m venv venv

CALL venv\Scripts\activate.bat

ECHO Installing packages

pip install -r requirements.txt
pip install -r requirements_dev.txt

ECHO setting up local development environment

pip install -e .

ECHO Finished

pause