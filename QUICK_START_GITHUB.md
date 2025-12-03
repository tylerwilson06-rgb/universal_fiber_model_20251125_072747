# Quick Start: Push to GitHub

## ⚠️ IMPORTANT: Git Must Be Installed First!

**Your system doesn't have Git installed yet.** Install it first:

### Install Git (Choose One):

1. **Git for Windows** (Command Line): https://git-scm.com/download/win
2. **GitHub Desktop** (Easier GUI): https://desktop.github.com/

**After installing, restart your terminal/PowerShell!**

---

## 🚀 Quick Method (Choose One)

### Option 1: Run PowerShell Script (Recommended)
```powershell
.\push_to_github.ps1
```

### Option 2: Run Batch File
```cmd
push_to_github.bat
```

### Option 3: Manual Commands (Copy & Paste)

```powershell
# 1. Initialize Git
git init

# 2. Add all files
git add .

# 3. Create commit
git commit -m "Initial commit: Production-ready Universal Fiber Sensor Model v1.0.0"

# 4. Add remote repository
git remote add origin https://github.com/tylerwilson06-rgb/universal-fiber-sensor-model.git

# 5. Set branch to main
git branch -M main

# 6. Push to GitHub
git push -u origin main
```

---

## 🔐 Authentication Required

When pushing, you'll need:

- **Username**: `tylerwilson06-rgb`
- **Password**: Use a **Personal Access Token** (NOT your GitHub password)

### Get Personal Access Token:
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Name it: "Universal Fiber Model"
4. Check `repo` scope
5. Click "Generate token"
6. **COPY THE TOKEN** (you won't see it again!)
7. Use this token as your password when prompted

---

## ✅ Verify Upload

After pushing, visit:
**https://github.com/tylerwilson06-rgb/universal-fiber-sensor-model**

You should see all your files!

---

## 📖 Full Guide

For detailed instructions and troubleshooting, see: `GITHUB_SETUP_GUIDE.md`

---

## Repository Info

- **Repository URL**: https://github.com/tylerwilson06-rgb/universal-fiber-sensor-model.git
- **Web URL**: https://github.com/tylerwilson06-rgb/universal-fiber-sensor-model
- **Owner**: tylerwilson06-rgb

