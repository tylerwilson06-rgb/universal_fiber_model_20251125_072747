# GitHub Setup Guide

## Quick Setup Instructions

### Prerequisites: Install Git First!

**Git is not currently installed on your system.** You need to install it first before pushing to GitHub.

#### Option 1: Download Git for Windows (Recommended)
1. Visit: https://git-scm.com/download/win
2. Download the installer
3. Run the installer (use default settings)
4. **Restart PowerShell/Command Prompt after installation**
5. Verify installation: `git --version`

#### Option 2: Install via GitHub Desktop (Easier GUI)
1. Visit: https://desktop.github.com/
2. Install GitHub Desktop
3. Sign in with your GitHub account
4. Clone or create repository through GUI

---

## Method 1: Using the PowerShell Script (Recommended)

### Step 1: Install Git (if not installed)
See prerequisites above.

### Step 2: Run the Script
```powershell
# Navigate to your project folder
cd "C:\Users\justt\Downloads\universal_fiber_model_20251125_072747"

# Run the script
.\push_to_github.ps1
```

The script will:
- ✅ Check if Git is installed
- ✅ Initialize repository (if needed)
- ✅ Add all files
- ✅ Create commit
- ✅ Add remote repository
- ✅ Push to GitHub

### Step 3: Authenticate When Prompted

When pushing, you'll need to authenticate:

**Option A: Personal Access Token (Recommended)**
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name: "Universal Fiber Model Upload"
4. Select scopes: Check `repo` (full control)
5. Click "Generate token"
6. **COPY THE TOKEN** (you won't see it again!)
7. When prompted for password, paste the token

**Option B: GitHub Desktop**
- Use GitHub Desktop GUI instead (much easier!)

---

## Method 2: Manual Commands (If Script Doesn't Work)

### Step 1: Open PowerShell in Project Folder
```powershell
cd "C:\Users\justt\Downloads\universal_fiber_model_20251125_072747"
```

### Step 2: Initialize Git
```powershell
git init
```

### Step 3: Add All Files
```powershell
git add .
```

### Step 4: Create Initial Commit
```powershell
git commit -m "Initial commit: Production-ready Universal Fiber Sensor Model v1.0.0"
```

### Step 5: Add Remote Repository
```powershell
git remote add origin https://github.com/tylerwilson06-rgb/universal-fiber-sensor-model.git
```

### Step 6: Set Branch to Main
```powershell
git branch -M main
```

### Step 7: Push to GitHub
```powershell
git push -u origin main
```

**When prompted:**
- Username: `tylerwilson06-rgb`
- Password: Use Personal Access Token (see authentication instructions above)

---

## Method 3: Using GitHub Desktop (Easiest for Beginners)

### Step 1: Install GitHub Desktop
1. Download from: https://desktop.github.com/
2. Install and sign in

### Step 2: Add Repository
1. Open GitHub Desktop
2. Click "File" → "Add Local Repository"
3. Browse to: `C:\Users\justt\Downloads\universal_fiber_model_20251125_072747`
4. Click "Add repository"

### Step 3: Publish Repository
1. Click "Publish repository" button
2. Repository name: `universal-fiber-sensor-model`
3. Owner: `tylerwilson06-rgb`
4. Description: "Universal Fiber Sensor Model - Production-ready implementation"
5. Make sure "Keep this code private" is unchecked (if you want it public)
6. Click "Publish repository"

---

## Troubleshooting

### Issue: "git is not recognized"
**Solution**: Git is not installed. Install from https://git-scm.com/download/win and restart PowerShell.

### Issue: "Authentication failed"
**Solution**: 
- Use Personal Access Token instead of password
- Get token from: https://github.com/settings/tokens
- Make sure token has `repo` scope

### Issue: "Repository not empty"
**Solution**: The GitHub repository might have files. You have two options:
- Option 1: Delete all files in GitHub repository first (through web interface)
- Option 2: Force push: `git push -u origin main --force` (BE CAREFUL - overwrites remote)

### Issue: "Remote origin already exists"
**Solution**: Remove and re-add:
```powershell
git remote remove origin
git remote add origin https://github.com/tylerwilson06-rgb/universal-fiber-sensor-model.git
```

### Issue: "Nothing to commit"
**Solution**: Files might already be committed. Check status:
```powershell
git status
```

---

## Verification

After pushing, verify your code is on GitHub:

1. Visit: https://github.com/tylerwilson06-rgb/universal-fiber-sensor-model
2. You should see all your files
3. Check that `README.md`, `src/`, `examples/`, etc. are there

---

## Repository Information

- **Repository URL**: https://github.com/tylerwilson06-rgb/universal-fiber-sensor-model.git
- **GitHub Web URL**: https://github.com/tylerwilson06-rgb/universal-fiber-sensor-model
- **Owner**: tylerwilson06-rgb
- **Branch**: main

---

## Next Steps After Uploading

1. ✅ Verify all files are uploaded correctly
2. ✅ Check README.md displays properly on GitHub
3. ✅ Add repository description and topics (optional)
4. ✅ Consider adding a LICENSE file
5. ✅ Add repository to your GitHub profile (optional)

---

## Need Help?

- Git Documentation: https://git-scm.com/doc
- GitHub Help: https://docs.github.com/en/get-started
- GitHub Support: https://support.github.com/

