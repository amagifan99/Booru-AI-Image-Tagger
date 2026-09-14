@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo =================================
echo       Python Script Launcher
echo =================================
echo.

REM =================================
REM CONFIGURATION
REM =================================

set "ROOT=%~dp0"

set "PYTHON_DIR=%ROOT%python"
set "PYTHON=%PYTHON_DIR%\python.exe"

set "GIT_CURL=%ROOT%curl\bin\curl.exe"

set "PYTHON_URL=https://www.python.org/ftp/python/3.10.5/python-3.10.5-embed-amd64.zip"
set "GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py"

set "PYTHON_ZIP=%ROOT%python.zip"
set "GET_PIP=%ROOT%get-pip.py"

set "MODEL_DIR=%ROOT%models\deepdanbooru-v3-20211112-sgd-e28"
set "MODEL_ZIP=%ROOT%deepdanbooru.zip"
set "MODEL_URL=https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip"

REM =================================
REM FFMPEG
REM =================================

REM FFmpeg is installed directly beside this BAT file
set "FFMPEG=%ROOT%ffmpeg.exe"

set "FFMPEG_ARCHIVE=%ROOT%ffmpeg.7z"
set "FFMPEG_TEMP=%ROOT%ffmpeg_temp"

set "FFMPEG_URL=https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-essentials.7z"

REM Portable 7-Zip extractor
set "SEVENZIP=%ROOT%7zr.exe"
set "SEVENZIP_URL=https://www.7-zip.org/a/7zr.exe"


REM =================================
REM CHECK CURL
REM =================================

if not exist "%GIT_CURL%" (
    echo ERROR: Portable curl was not found!
    echo.
    echo Expected:
    echo %GIT_CURL%
    echo.
    pause
    exit /b 1
)


REM =================================
REM DOWNLOAD + INSTALL PYTHON
REM =================================

if not exist "%PYTHON%" (

    echo Python 3.10.5 was not found.
    echo.
    echo Downloading portable Python 3.10.5...
    echo.

    "%GIT_CURL%" -L --fail --progress-bar ^
        -o "%PYTHON_ZIP%" ^
        "%PYTHON_URL%"

    if errorlevel 1 (
        echo.
        echo ERROR: Failed to download Python.
        pause
        exit /b 1
    )

    echo.
    echo Extracting Python...

    if exist "%PYTHON_DIR%" (
        rmdir /s /q "%PYTHON_DIR%"
    )

    mkdir "%PYTHON_DIR%"

    powershell -NoProfile -Command ^
        "Expand-Archive -LiteralPath '%PYTHON_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"

    if errorlevel 1 (
        echo.
        echo ERROR: Failed to extract Python.
        pause
        exit /b 1
    )

    del "%PYTHON_ZIP%" >nul 2>&1

    echo Python extracted successfully.
    echo.


    REM =================================
    REM ENABLE SITE-PACKAGES
    REM =================================

    echo Configuring Python...

    powershell -NoProfile -Command ^
        "$p='%PYTHON_DIR%\python310._pth'; $c=Get-Content $p; if ($c -notcontains 'import site') { Add-Content $p 'import site' }"

    if errorlevel 1 (
        echo.
        echo ERROR: Failed to configure Python.
        pause
        exit /b 1
    )


    REM =================================
    REM INSTALL PIP
    REM =================================

    echo Downloading pip installer...

    "%GIT_CURL%" -L --fail --progress-bar ^
        -o "%GET_PIP%" ^
        "%GET_PIP_URL%"

    if errorlevel 1 (
        echo.
        echo ERROR: Failed to download get-pip.py.
        pause
        exit /b 1
    )

    echo.
    echo Installing pip...

    "%PYTHON%" "%GET_PIP%"

    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install pip.
        pause
        exit /b 1
    )

    del "%GET_PIP%" >nul 2>&1

    echo.
    echo Python setup complete.
    echo.
)


REM =================================
REM SHOW PYTHON VERSION
REM =================================

echo Using portable Python:
"%PYTHON%" --version
echo.


REM =================================
REM DOWNLOAD MODEL
REM =================================

if not exist "%MODEL_DIR%\project.json" (

    echo DeepDanbooru model was not found.
    echo.
    echo Downloading model... deepdanbooru-v3-20211112-sgd-e28.zip  [571MB]
    echo.

    "%GIT_CURL%" -L --fail --progress-bar ^
        -o "%MODEL_ZIP%" ^
        "%MODEL_URL%"

    if errorlevel 1 (
        echo.
        echo ERROR: Failed to download DeepDanbooru model.
        pause
        exit /b 1
    )

    echo.
    echo Extracting DeepDanbooru model...

    if not exist "%ROOT%models" mkdir "%ROOT%models"
    if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

    powershell -NoProfile -Command ^
        "Expand-Archive -LiteralPath '%MODEL_ZIP%' -DestinationPath '%MODEL_DIR%' -Force"

    if errorlevel 1 (
        echo.
        echo ERROR: Failed to extract DeepDanbooru model.
        pause
        exit /b 1
    )

    del "%MODEL_ZIP%" >nul 2>&1

    echo.
    echo DeepDanbooru model installed.
    echo.
)


REM =================================
REM INSTALL DEPENDENCIES
REM =================================

echo Installing dependencies...
echo.

"%PYTHON%" -m pip install -r "%ROOT%requirements.txt"

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies.
    echo.
    pause
    exit /b 1
)

echo.
echo Dependencies installed successfully.
echo.


REM =================================
REM MENU
REM =================================

:MENU

cls

echo =================================
echo       Script Launcher
echo =================================
echo.
echo Using
"%PYTHON%" --version
echo.
echo 1. Run Image Tagger
echo 2. Run Video Scanner
echo 3. Exit
echo.

choice /c 123 /n /m "Choose an option: "

if errorlevel 3 exit
if errorlevel 2 goto CHECK_FFMPEG
if errorlevel 1 goto AUTOTAGGER


REM =================================
REM IMAGE TAGGER
REM =================================

:AUTOTAGGER

echo.
echo Starting Image Tagger...
echo.

"%PYTHON%" "%ROOT%autotagger.py"

echo.
pause
goto MENU


REM =================================
REM CHECK FFMPEG
REM =================================

:CHECK_FFMPEG

echo.
echo Checking for FFmpeg...
echo.


REM ---------------------------------
REM Check system-wide FFmpeg
REM ---------------------------------

where ffmpeg >nul 2>&1

if not errorlevel 1 (
    echo FFmpeg found
    echo.
    goto VIDEO_SCANNER
)


REM ---------------------------------
REM Check portable FFmpeg beside BAT
REM ---------------------------------

if exist "%FFMPEG%" (
    echo Portable FFmpeg found beside launcher.
    echo.
    echo Location:
    echo %FFMPEG%
    echo.

    set "PATH=%ROOT%;%PATH%"

    goto VIDEO_SCANNER
)


REM =================================
REM FFMPEG IS MISSING
REM =================================

echo FFmpeg was not found.
echo.
echo The Video Scanner requires FFmpeg.
echo.

choice /c YN /n /m "Do you want to download FFmpeg now? (Y/N): "

if errorlevel 2 (
    echo.
    echo skipped FFMPEG
    echo Exporting clips may not work.
    echo.
    goto VIDEO_SCANNER
)

if errorlevel 1 goto DOWNLOAD_FFMPEG


REM =================================
REM DOWNLOAD FFMPEG
REM =================================

:DOWNLOAD_FFMPEG

echo.
echo =================================
echo       Downloading FFmpeg
echo =================================
echo.
echo Downloading FFmpeg...
echo.

"%GIT_CURL%" -L --fail --progress-bar ^
    -o "%FFMPEG_ARCHIVE%" ^
    "%FFMPEG_URL%"

if errorlevel 1 (
    echo.
    echo ERROR: Failed to download FFmpeg.
    echo.

    if exist "%FFMPEG_ARCHIVE%" (
        del "%FFMPEG_ARCHIVE%" >nul 2>&1
    )

    pause
    goto MENU
)

echo.
echo FFmpeg download complete.
echo.


REM =================================
REM DOWNLOAD 7-ZIP EXTRACTOR
REM =================================

if not exist "%SEVENZIP%" (

    echo Downloading portable 7-Zip extractor...
    echo.

    "%GIT_CURL%" -L --fail --progress-bar ^
        -o "%SEVENZIP%" ^
        "%SEVENZIP_URL%"

    if errorlevel 1 (
        echo.
        echo ERROR: Failed to download 7-Zip extractor.
        echo.

        if exist "%FFMPEG_ARCHIVE%" (
            del "%FFMPEG_ARCHIVE%" >nul 2>&1
        )

        if exist "%SEVENZIP%" (
            del "%SEVENZIP%" >nul 2>&1
        )

        pause
        goto MENU
    )
)


REM =================================
REM EXTRACT FFMPEG
REM =================================

echo.
echo Extracting FFmpeg...
echo.

if exist "%FFMPEG_TEMP%" (
    rmdir /s /q "%FFMPEG_TEMP%"
)

mkdir "%FFMPEG_TEMP%"

"%SEVENZIP%" x "%FFMPEG_ARCHIVE%" -o"%FFMPEG_TEMP%" -y

if errorlevel 1 (
    echo.
    echo ERROR: Failed to extract FFmpeg.
    echo.

    rmdir /s /q "%FFMPEG_TEMP%" >nul 2>&1
    del "%FFMPEG_ARCHIVE%" >nul 2>&1

    pause
    goto MENU
)

echo.
echo FFmpeg archive extracted successfully.
echo.


REM =================================
REM FIND FFMPEG.EXE IN BIN DIRECTORY
REM =================================

echo.
echo Searching for FFmpeg in extracted bin directory...
echo.

set "FOUND_FFMPEG="
set "FOUND_BIN="

for /r "%FFMPEG_TEMP%" %%F in (ffmpeg.exe) do (
    if /i "%%~nxF"=="ffmpeg.exe" (
        for %%D in ("%%~dpF.") do (
            if /i "%%~nxD"=="bin" (
                set "FOUND_FFMPEG=%%~fF"
                set "FOUND_BIN=%%~dpF"
            )
        )
    )
)

if not defined FOUND_FFMPEG (
    echo.
    echo ERROR: Could not find:
    echo.
    echo %FFMPEG_TEMP%\*\bin\ffmpeg.exe
    echo.
    echo Extracted files found:
    dir /s /b "%FFMPEG_TEMP%\ffmpeg.exe" 2>nul
    echo.
    pause
    goto MENU
)

echo.
echo =================================
echo       FFmpeg Found
echo =================================
echo.
echo Executable:
echo !FOUND_FFMPEG!
echo.
echo Bin directory:
echo !FOUND_BIN!
echo.


REM =================================
REM INSTALL FFMPEG BESIDE BAT
REM =================================

echo Installing FFmpeg beside launcher...
echo.

REM Copy ffmpeg.exe
copy /Y "!FOUND_BIN!ffmpeg.exe" "%ROOT%ffmpeg.exe"

if errorlevel 1 (
    echo.
    echo ERROR: Failed to copy ffmpeg.exe.
    echo.
    echo Source:
    echo !FOUND_BIN!ffmpeg.exe
    echo.
    echo Destination:
    echo %ROOT%ffmpeg.exe
    echo.
    pause
    goto MENU
)




REM =================================
REM VERIFY INSTALLATION
REM =================================

echo.
echo Checking installation...
echo.

if not exist "%ROOT%ffmpeg.exe" (
    echo.
    echo ERROR: ffmpeg.exe was not installed.
    echo.
    pause
    goto MENU
)

echo.
echo =================================
echo       FFmpeg Installed
echo =================================
echo.
echo FFmpeg:
echo %ROOT%ffmpeg.exe
echo.


REM =================================
REM CLEANUP
REM =================================

rmdir /s /q "%FFMPEG_TEMP%" >nul 2>&1
del "%FFMPEG_ARCHIVE%" >nul 2>&1
del "%SEVENZIP%" >nul 2>&1


REM =================================
REM ADD BAT DIRECTORY TO PATH
REM =================================

set "PATH=%ROOT%;%PATH%"


REM =================================
REM TEST FFMPEG
REM =================================

echo Testing FFmpeg...
echo.

"%ROOT%ffmpeg.exe" -version 2>&1 | findstr /i "ffmpeg version"

if errorlevel 1 (
    echo.
    echo ERROR: FFmpeg was installed but could not run.
    echo.
    pause
    goto MENU
)

echo.
echo FFmpeg is ready.
echo.
goto VIDEO_SCANNER


REM =================================
REM VIDEO SCANNER
REM =================================

:VIDEO_SCANNER

echo.
echo Starting Video Scanner...
echo.

"%PYTHON%" "%ROOT%video_analyzer.py"

echo.
pause
goto MENU
