@echo off
REM Batch Script to Push Code to GitHub
REM Repository: https://github.com/tylerwilson06-rgb/universal-fiber-sensor-model.git

echo ===========================================
echo GitHub Upload Script
echo Repository: universal-fiber-sensor-model
echo ===========================================
echo.

REM Check if git is installed
echo Step 1: Checking if Git is installed...
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Git is not installed or not in PATH
    echo.
    echo Please install Git from: https://git-scm.com/download/win
    echo After installing, restart Command Prompt and run this script again.
    pause
    exit /b 1
)
git --version
echo Git found!
echo.

REM Step 2: Initialize git repository
echo Step 2: Initializing Git repository...
if exist .git (
    echo Git repository already initialized
) else (
    git init
    if %errorlevel% neq 0 (
        echo ERROR: Failed to initialize git repository
        pause
        exit /b 1
    )
    echo Git repository initialized
)
echo.

REM Step 3: Add all files
echo Step 3: Adding all files to staging...
git add .
if %errorlevel% neq 0 (
    echo ERROR: Failed to add files
    pause
    exit /b 1
)
echo All files added to staging
echo.

REM Step 4: Create initial commit
echo Step 4: Creating initial commit...
git commit -m "Initial commit: Production-ready Universal Fiber Sensor Model v1.0.0"
if %errorlevel% neq 0 (
    echo Warning: Commit skipped (might already be committed)
)
echo.

REM Step 5: Add remote repository
echo Step 5: Adding remote repository...
git remote remove origin >nul 2>&1
git remote add origin https://github.com/tylerwilson06-rgb/universal-fiber-sensor-model.git
if %errorlevel% neq 0 (
    echo ERROR: Failed to add remote repository
    pause
    exit /b 1
)
echo Remote 'origin' added
echo.

REM Step 6: Set branch to main
echo Step 6: Setting default branch to 'main'...
git branch -M main
echo Default branch set to 'main'
echo.

REM Step 7: Push to GitHub
echo Step 7: Pushing to GitHub...
echo.
echo NOTE: You may be prompted for GitHub credentials.
echo       Username: tylerwilson06-rgb
echo       Password: Use Personal Access Token (not your GitHub password)
echo       Get token: https://github.com/settings/tokens
echo.
pause
echo.

git push -u origin main
if %errorlevel% equ 0 (
    echo.
    echo ===========================================
    echo SUCCESS! Code pushed to GitHub!
    echo ===========================================
    echo.
    echo Your repository is now available at:
    echo https://github.com/tylerwilson06-rgb/universal-fiber-sensor-model
    echo.
) else (
    echo.
    echo ===========================================
    echo Push failed. Possible reasons:
    echo ===========================================
    echo.
    echo 1. Authentication failed
    echo    - Use Personal Access Token: https://github.com/settings/tokens
    echo.
    echo 2. Repository is not empty
    echo    - Delete files in GitHub repository first, or
    echo    - Use: git push -u origin main --force (BE CAREFUL!)
    echo.
    echo 3. Network connection issue
    echo    - Check your internet connection
    echo.
)

pause





