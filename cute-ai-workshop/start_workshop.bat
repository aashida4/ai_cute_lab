@echo off
setlocal
title Cute AI Workshop
pushd "%~dp0"
if errorlevel 1 (
    echo Could not open the workshop folder.
    pause
    exit /b 1
)

set "UV_EXE=uv"
where uv >nul 2>nul
if errorlevel 1 (
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
    ) else (
        echo uv was not found. Please ask your instructor to complete setup.
        popd
        pause
        exit /b 1
    )
)

echo Starting JupyterLab. Keep this window open while working.
call "%UV_EXE%" run --locked jupyter lab
set "RESULT=%ERRORLEVEL%"
popd
if not "%RESULT%"=="0" (
    echo JupyterLab exited with an error. Please show this window to your instructor.
    pause
)
exit /b %RESULT%
