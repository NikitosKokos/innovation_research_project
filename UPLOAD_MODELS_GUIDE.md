# Guide: Uploading Large AI Models to GitHub

Your AI models (like `ft_model.pth`) are larger than GitHub's 100MB limit. To include them in your repository, you must use **Git LFS (Large File Storage)**.

## 🚀 Step-by-Step Instructions

### 1. Install Git LFS
If you haven't already, install Git LFS on your machine:
- **Windows**: Download and run the installer from [git-lfs.github.com](https://git-lfs.github.com/).
- **macOS**: `brew install git-lfs`
- **Linux**: `sudo apt install git-lfs`

### 2. Initialize LFS in your Project
Open your terminal in the project root and run:
```bash
git lfs install
```

### 3. Track Model Files
Tell Git LFS which files to handle. We have already prepared a `.gitattributes` file for you, but you can manually track extensions like this:
```bash
git lfs track "*.pth"
git lfs track "*.onnx"
git lfs track "*.trt"
```

### 4. Update .gitignore (Crucial)
Your current `.gitignore` is set to ignore `AI_models/` and `*.pth`. You must remove these lines or force add the files so Git LFS can track them.
- Open `.gitignore` and remove:
  ```text
  runs/
  AI_models/
  *.pth
  ```

### 5. Commit and Push
Now you can add and push your models just like regular files:
```bash
git add .gitattributes
git add AI_models/
git commit -m "add: fine-tuned AI models using Git LFS"
git push origin main
```

---

## ⚠️ Important Notes
- **Storage Limits**: Free GitHub accounts have a 1GB storage limit for Git LFS. Since your models are ~400MB each, you can store about 2-3 models before hitting the limit.
- **Bandwidth**: Git LFS also has a monthly bandwidth limit (1GB for free accounts).
- **Cloning**: When someone else clones this repo, they will need to have Git LFS installed to download the actual model files instead of just pointers.

## 🛠️ Alternative: Hugging Face (Recommended for large files)
If you have many heavy models, consider uploading them to [Hugging Face](https://huggingface.co/) and using the `hf_utils.py` script in this repo to download them automatically. This is what many professional AI projects do to bypass GitHub's limits.
