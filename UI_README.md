# Pixel Perfect - Streamlit UI

A visual pipeline builder for sophisticated image processing operations.

## 🚀 Quick Start

### Option 1: Using the launch script (Recommended)
```bash
python run_ui.py
```

### Option 2: Direct Streamlit command
```bash
cd src
uv run streamlit run ui/streamlit_app.py
```

### Option 3: Using the console script
```bash
uv run pixel-perfect-ui
```

The app will be available at: http://localhost:8501

## 📋 Current Features (Task 016)

✅ **Basic App Structure**
- Clean Streamlit interface with sidebar and main content area
- Session state management for maintaining application state
- Responsive layout that adapts to different screen sizes

✅ **Image Upload & Display**
- Drag-and-drop image upload with format validation
- Support for PNG, JPG, JPEG, WebP, BMP, TIFF formats
- Image preview and metadata display
- Optimized image display for web performance

✅ **Basic Pipeline Integration**
- Integration with existing Pipeline and Operation classes
- Simple pipeline execution with progress feedback
- Error handling and user-friendly error messages
- Side-by-side image comparison

✅ **Session Management**
- Persistent state across UI interactions
- Reset functions for pipeline and entire application
- Pipeline summary and status indicators

## 🎯 Usage

1. **Upload an Image**: Use the sidebar file uploader to select an image
2. **Add Test Operation**: Click "🧪 Add Test Operation" to add a sample PixelFilter
3. **Execute Pipeline**: Click "▶️ Execute Pipeline" to process the image
4. **View Results**: See the processed image alongside the original

## 🔧 Current Limitations

This is the foundation implementation (Task 016). The following features are coming in subsequent tasks:

- **Operation Selector** (Task 017): Proper operation browsing and selection
- **Parameter Configuration** (Task 018): Dynamic parameter forms with validation
- **Real-time Execution** (Task 019): Live processing as parameters change
- **Advanced Image Display** (Task 020): Zoom, pan, comparison tools
- **Export & Sharing** (Task 021): Save/load pipelines and export results

## 🏗️ Architecture

```
src/ui/
├── streamlit_app.py           # Main application entry point
├── components/
│   ├── session.py             # Session state management
│   ├── layout.py              # App layout components
│   └── image_viewer.py        # Image upload and display
└── utils/
    └── image_utils.py         # Image processing utilities
```

## 🐛 Troubleshooting

**App won't start:**
- Make sure you're in the project root directory
- Ensure all dependencies are installed: `uv sync`
- Check that Streamlit is available: `uv run streamlit --version`

**Image upload fails:**
- Check file size (max 50MB)
- Ensure file is a valid image format
- Try a different image to isolate the issue

**Pipeline execution fails:**
- Check that operations have valid parameters
- Look for error messages in the UI
- Some operations may not work with certain image types

## 🔍 Testing

The current implementation includes:
- ✅ Image upload validation
- ✅ Basic pipeline execution
- ✅ Error handling and display
- ✅ Session state management

## 📝 Development Notes

This foundation provides:
- Solid architecture for building advanced features
- Integration with existing Pipeline/Operation framework
- Clean separation of concerns between components
- Extensible design for future enhancements

The next task (017) will add the operation selector interface to make the UI much more powerful and user-friendly.
