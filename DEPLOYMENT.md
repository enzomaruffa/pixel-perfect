# 🚀 Streamlit Cloud Deployment Guide

This guide walks you through deploying Pixel Perfect to Streamlit Cloud.

## 📋 Prerequisites

1. **GitHub Account**: Your code must be in a GitHub repository
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)

## 📁 Required Files (Already Created)

The following files are required for Streamlit Cloud deployment and have been created for you:

### 1. `streamlit_app.py` (Root Entry Point)
- **Location**: Root directory
- **Purpose**: Main entry point that Streamlit Cloud looks for
- **Content**: Imports and runs the main app from `src/ui/streamlit_app.py`

### 2. `requirements.txt`
- **Location**: Root directory
- **Purpose**: Lists all Python dependencies
- **Content**: Core packages like Streamlit, NumPy, Pillow, Pydantic

### 3. `packages.txt`
- **Location**: Root directory
- **Purpose**: System-level dependencies for image processing
- **Content**: Graphics libraries needed for PIL/Pillow

### 4. `.streamlit/config.toml`
- **Location**: `.streamlit/` directory
- **Purpose**: Streamlit app configuration
- **Content**: Performance optimizations and theme settings

## 🔧 Deployment Steps

### Step 1: Push to GitHub
```bash
# Make sure all files are committed
git add .
git commit -m "Add Streamlit deployment configuration"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**: Visit [share.streamlit.io](https://share.streamlit.io)

2. **Sign in**: Use your GitHub account

3. **Create New App**: Click "New app"

4. **Repository Selection**:
   - **Repository**: Select your `pixel-perfect` repository
   - **Branch**: Choose `main` (or your default branch)
   - **Main file path**: Leave as `streamlit_app.py` (default)

5. **Advanced Settings** (Optional):
   - **Python version**: 3.10+ (recommended)
   - **Environment variables**: None needed

6. **Deploy**: Click "Deploy!"

### Step 3: Wait for Deployment
- Initial deployment takes 2-5 minutes
- You'll see logs showing package installation
- App will automatically start once complete

### Step 4: Access Your App
- You'll get a URL like: `https://your-app-name.streamlit.app`
- Share this URL with others!

## 🛠️ Troubleshooting

### Common Issues

**1. Import Errors**
```
ModuleNotFoundError: No module named 'operations'
```
- **Solution**: The `streamlit_app.py` automatically adds `src/` to Python path
- **Check**: Make sure `streamlit_app.py` is in the root directory

**2. Missing Dependencies**
```
ModuleNotFoundError: No module named 'xyz'
```
- **Solution**: Add the missing package to `requirements.txt`
- **Example**: If you need `matplotlib`, add `matplotlib>=3.7.0`

**3. Image Processing Errors**
```
ImportError: cannot import name '_imaging' from 'PIL'
```
- **Solution**: The `packages.txt` file includes necessary system libraries
- **Check**: Ensure `packages.txt` is in the root directory

**4. Memory Issues**
```
Streamlit app crashed due to memory limit
```
- **Solution**: Reduce image sizes or optimize operations
- **Note**: Streamlit Cloud has memory limits (~1GB)

### Debug Steps

1. **Check Logs**: In Streamlit Cloud, click "Manage app" → "Logs"
2. **Test Locally**: Run `streamlit run streamlit_app.py` locally first
3. **Validate Setup**: Run `python deployment_test.py` to test imports

## ⚡ Performance Optimization

### For Better Performance on Streamlit Cloud:

1. **Enable Caching** (Already configured):
   ```python
   @st.cache_data
   def process_image(image, operations):
       # Your processing code
   ```

2. **Optimize Images**:
   - Resize large images before processing
   - Use appropriate formats (WebP for smaller files)

3. **Limit Pipeline Complexity**:
   - Start with simple operations
   - Monitor memory usage

4. **Use Session State Efficiently**:
   - Store only necessary data
   - Clear unused results

## 🔄 Updating Your Deployment

To update your deployed app:

1. **Make Changes**: Modify your code locally
2. **Test**: Run `streamlit run streamlit_app.py` to test
3. **Commit & Push**:
   ```bash
   git add .
   git commit -m "Update: description of changes"
   git push origin main
   ```
4. **Auto-Deploy**: Streamlit Cloud automatically redeploys on push

## 🎯 App Features Available

Once deployed, users can:

- ✅ Upload images (up to 200MB)
- ✅ Build processing pipelines visually
- ✅ Use formula-based operations
- ✅ Real-time preview and execution
- ✅ Export processed images
- ✅ Save and load pipeline configurations

## 🔐 Security Notes

- **No Authentication**: App is publicly accessible
- **Safe Expressions**: Formula evaluation is sandboxed
- **File Handling**: Only image files are accepted
- **No Persistence**: User data is not stored between sessions

## 📱 Mobile Experience

The app is optimized for desktop but works on mobile:
- **Responsive Layout**: Adapts to screen size
- **Touch Interface**: Full touch support
- **Sidebar**: Collapsible on mobile

## 🎉 You're Ready!

Your Pixel Perfect app should now be live and accessible to anyone with the URL. Share it with:

- **Designers**: For creative image effects
- **Developers**: As a demo of pipeline architecture
- **Students**: For learning image processing
- **Artists**: For experimental digital art

---

**Need Help?** Check the Streamlit Cloud [documentation](https://docs.streamlit.io/streamlit-community-cloud) or the app logs for detailed error messages.
