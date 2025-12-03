# PowerShell Script to Push Code to GitHub
# Repository: https://github.com/tylerwilson06-rgb/universal-fiber-sensor-model.git

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "GitHub Upload Script" -ForegroundColor Cyan
Write-Host "Repository: universal-fiber-sensor-model" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
Write-Host "Step 1: Checking if Git is installed..." -ForegroundColor Yellow
try {
    $gitVersion = git --version
    Write-Host "✓ Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ ERROR: Git is not installed or not in PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "After installing, restart PowerShell and run this script again." -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Step 2: Initialize git repository (if not already initialized)
Write-Host "Step 2: Initializing Git repository..." -ForegroundColor Yellow
if (Test-Path ".git") {
    Write-Host "✓ Git repository already initialized" -ForegroundColor Green
} else {
    git init
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Git repository initialized" -ForegroundColor Green
    } else {
        Write-Host "✗ ERROR: Failed to initialize git repository" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Step 3: Add all files
Write-Host "Step 3: Adding all files to staging..." -ForegroundColor Yellow
git add .
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ All files added to staging" -ForegroundColor Green
    $stagedFiles = (git status --short | Measure-Object -Line).Lines
    Write-Host "  Files staged: $stagedFiles" -ForegroundColor Gray
} else {
    Write-Host "✗ ERROR: Failed to add files" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 4: Check if there are changes to commit
Write-Host "Step 4: Checking for changes to commit..." -ForegroundColor Yellow
$status = git status --short
if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Host "⚠ No changes to commit. Repository is up to date." -ForegroundColor Yellow
    Write-Host "  If you want to force push, you may need to check remote status." -ForegroundColor Yellow
} else {
    Write-Host "✓ Changes detected, ready to commit" -ForegroundColor Green
}

Write-Host ""

# Step 5: Create initial commit
Write-Host "Step 5: Creating initial commit..." -ForegroundColor Yellow
$commitMessage = "Initial commit: Production-ready Universal Fiber Sensor Model v1.0.0

- Complete codebase with robust error handling
- Universal Feature Vector (UFV) extraction with 204 features
- Multi-head neural network for event classification, risk assessment, damage detection
- Production-ready inference interface with GPU support and batch processing
- Comprehensive documentation and examples
- Compatible with trained_model.pth"

git commit -m $commitMessage
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Commit created successfully" -ForegroundColor Green
} else {
    Write-Host "⚠ Commit skipped (no changes or already committed)" -ForegroundColor Yellow
}

Write-Host ""

# Step 6: Add remote repository
Write-Host "Step 6: Adding remote repository..." -ForegroundColor Yellow
$remoteUrl = "https://github.com/tylerwilson06-rgb/universal-fiber-sensor-model.git"
$remoteName = "origin"

# Check if remote already exists
$existingRemote = git remote get-url $remoteName 2>$null
if ($LASTEXITCODE -eq 0) {
    if ($existingRemote -eq $remoteUrl) {
        Write-Host "✓ Remote 'origin' already configured correctly" -ForegroundColor Green
    } else {
        Write-Host "⚠ Remote 'origin' exists with different URL: $existingRemote" -ForegroundColor Yellow
        Write-Host "  Removing old remote and adding new one..." -ForegroundColor Yellow
        git remote remove $remoteName
        git remote add $remoteName $remoteUrl
        Write-Host "✓ Remote updated" -ForegroundColor Green
    }
} else {
    git remote add $remoteName $remoteUrl
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Remote 'origin' added: $remoteUrl" -ForegroundColor Green
    } else {
        Write-Host "✗ ERROR: Failed to add remote repository" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Step 7: Set default branch to main
Write-Host "Step 7: Setting default branch to 'main'..." -ForegroundColor Yellow
git branch -M main
Write-Host "✓ Default branch set to 'main'" -ForegroundColor Green

Write-Host ""

# Step 8: Push to GitHub
Write-Host "Step 8: Pushing to GitHub..." -ForegroundColor Yellow
Write-Host "  Repository: $remoteUrl" -ForegroundColor Gray
Write-Host "  Branch: main" -ForegroundColor Gray
Write-Host ""
Write-Host "⚠ NOTE: You may be prompted for GitHub credentials." -ForegroundColor Yellow
Write-Host "   You can use:" -ForegroundColor Yellow
Write-Host "   - Personal Access Token (recommended)" -ForegroundColor Yellow
Write-Host "   - GitHub Desktop app" -ForegroundColor Yellow
Write-Host "   - SSH key authentication" -ForegroundColor Yellow
Write-Host ""
Write-Host "   Get Personal Access Token: https://github.com/settings/tokens" -ForegroundColor Cyan
Write-Host ""

$pushChoice = Read-Host "Ready to push? (y/n)"
if ($pushChoice -eq "y" -or $pushChoice -eq "Y") {
    Write-Host ""
    Write-Host "Pushing code to GitHub..." -ForegroundColor Yellow
    
    # Try to push
    git push -u origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "===========================================" -ForegroundColor Green
        Write-Host "✓ SUCCESS! Code pushed to GitHub!" -ForegroundColor Green
        Write-Host "===========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Your repository is now available at:" -ForegroundColor Cyan
        Write-Host "https://github.com/tylerwilson06-rgb/universal-fiber-sensor-model" -ForegroundColor Cyan
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "===========================================" -ForegroundColor Red
        Write-Host "✗ Push failed. Possible reasons:" -ForegroundColor Red
        Write-Host "===========================================" -ForegroundColor Red
        Write-Host ""
        Write-Host "1. Authentication failed" -ForegroundColor Yellow
        Write-Host "   Solution: Use Personal Access Token or configure SSH" -ForegroundColor Gray
        Write-Host ""
        Write-Host "2. Repository is not empty (if you want to force push)" -ForegroundColor Yellow
        Write-Host "   Solution: Use 'git push -u origin main --force' (BE CAREFUL!)" -ForegroundColor Gray
        Write-Host ""
        Write-Host "3. Network connection issue" -ForegroundColor Yellow
        Write-Host "   Solution: Check your internet connection" -ForegroundColor Gray
        Write-Host ""
        Write-Host "For help, visit: https://docs.github.com/en/get-started/getting-started-with-git" -ForegroundColor Cyan
    }
} else {
    Write-Host ""
    Write-Host "Push cancelled. You can run this script again later." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To push manually, run:" -ForegroundColor Cyan
    Write-Host "  git push -u origin main" -ForegroundColor Gray
}

Write-Host ""

