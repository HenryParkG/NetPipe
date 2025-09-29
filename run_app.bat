@echo off
REM Activate Python virtual environment
call ".\npvenv\Scripts\activate.bat"

REM Run Streamlit app
streamlit run main.py

REM Pause to keep window open after exit
pause