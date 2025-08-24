# Task 021: Export and Sharing System

## Objective
Create comprehensive export and sharing capabilities that allow users to save processed images, export pipeline configurations, generate processing reports, and share their work with others.

## Requirements
1. Export processed images in multiple formats and qualities
2. Save and load pipeline configurations
3. Generate processing reports and documentation
4. Create shareable URLs for pipeline configurations
5. Implement batch export for multiple results

## File Structure to Create
```
src/ui/components/
├── export_manager.py        # Main export functionality
├── sharing_system.py        # URL sharing and collaboration
├── report_generator.py      # Processing documentation
└── batch_export.py          # Bulk export operations

src/ui/utils/
├── image_export.py          # Image format conversion and optimization
├── config_serialization.py  # Pipeline configuration serialization
├── url_encoding.py          # URL parameter encoding/decoding
└── report_templates.py      # Report generation templates
```

## Core Features

### Image Export Options
- **Format Selection**: PNG, JPEG, WebP, TIFF with quality settings
- **Size Options**: Original size, web-optimized, custom dimensions
- **Naming Conventions**: Automatic or custom file naming
- **Metadata Preservation**: Include/exclude EXIF and processing info
- **Batch Export**: Export all pipeline step results at once

### Pipeline Configuration Export
- **JSON Export**: Full pipeline configuration for reproducibility
- **YAML Export**: Human-readable pipeline format
- **Python Code Export**: Generate equivalent Python script
- **Preset Creation**: Save current pipeline as reusable preset

### Processing Reports
- **Visual Report**: Before/after comparisons with operation details
- **Technical Report**: Processing times, parameters, memory usage
- **Step-by-Step Guide**: Detailed operation breakdown
- **Performance Analysis**: Bottleneck identification and optimization suggestions

### Sharing and Collaboration
- **URL Sharing**: Encode pipeline configuration in shareable URLs
- **Preset Sharing**: Share custom presets with community
- **Gallery Integration**: Showcase processed images with settings
- **Version Control**: Track and manage pipeline iterations

## Technical Implementation

### Main Export Interface
```python
def render_export_section():
    """Main export and sharing interface"""
    if not st.session_state.processed_image:
        st.info("Process an image first to enable export options")
        return

    st.header("Export & Share")

    tab1, tab2, tab3, tab4 = st.tabs(["Images", "Pipeline", "Report", "Share"])

    with tab1:
        render_image_export_options()

    with tab2:
        render_pipeline_export_options()

    with tab3:
        render_report_generation()

    with tab4:
        render_sharing_options()

def render_image_export_options():
    """Image export configuration and download"""
    st.subheader("Image Export")

    # Export format selection
    export_format = st.selectbox(
        "Format",
        ["PNG", "JPEG", "WebP", "TIFF"],
        help="Choose output image format"
    )

    # Quality settings for lossy formats
    if export_format in ["JPEG", "WebP"]:
        quality = st.slider("Quality", 1, 100, 95)
    else:
        quality = 100

    # Size options
    size_option = st.radio(
        "Size",
        ["Original", "Web Optimized (1920px)", "Custom"],
        help="Choose output image dimensions"
    )

    if size_option == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            custom_width = st.number_input("Width", 1, 8000, 1920)
        with col2:
            custom_height = st.number_input("Height", 1, 8000, 1080)

    # Filename
    filename = st.text_input(
        "Filename",
        value=generate_default_filename(export_format),
        help="Enter filename without extension"
    )

    # Export button
    if st.button("Export Image", type="primary"):
        export_processed_image(
            format=export_format,
            quality=quality,
            size_option=size_option,
            filename=filename
        )
```

### Pipeline Configuration Export
```python
def export_pipeline_config(format_type="json"):
    """Export current pipeline configuration"""
    config = {
        "pipeline_info": {
            "created": datetime.now().isoformat(),
            "version": "1.0",
            "original_image_info": get_image_info(st.session_state.original_image)
        },
        "operations": []
    }

    # Add each operation to config
    for i, operation in enumerate(st.session_state.pipeline_operations):
        operation_config = {
            "step": i + 1,
            "operation": operation['name'],
            "parameters": operation['params'],
            "enabled": operation.get('enabled', True)
        }
        config["operations"].append(operation_config)

    if format_type == "json":
        return json.dumps(config, indent=2)
    elif format_type == "yaml":
        return yaml.dump(config, default_flow_style=False)
    elif format_type == "python":
        return generate_python_code(config)

def generate_python_code(config):
    """Generate equivalent Python code for pipeline"""
    code_lines = [
        "from src import Pipeline",
        "from src.operations import *",
        "",
        "# Generated pipeline code",
        f"pipeline = Pipeline('input_image.jpg')",
        ""
    ]

    for op_config in config["operations"]:
        if op_config["enabled"]:
            operation_name = op_config["operation"]
            params = op_config["parameters"]

            # Format parameters as Python code
            param_str = ", ".join([
                f"{k}={repr(v)}" for k, v in params.items()
            ])

            code_lines.append(f"pipeline.add({operation_name}({param_str}))")

    code_lines.extend([
        "",
        "# Execute pipeline",
        "result = pipeline.execute('output_image.png')"
    ])

    return "\n".join(code_lines)
```

### Report Generation
```python
class ProcessingReportGenerator:
    def __init__(self):
        self.template_dir = Path("src/ui/templates")

    def generate_visual_report(self):
        """Generate visual processing report"""
        report = {
            "title": "Image Processing Report",
            "timestamp": datetime.now().isoformat(),
            "original_image": st.session_state.original_image,
            "processed_image": st.session_state.processed_image,
            "pipeline_steps": st.session_state.execution_results,
            "total_execution_time": self.calculate_total_time(),
            "memory_usage": self.calculate_memory_usage(),
            "cache_statistics": self.get_cache_stats()
        }

        return self.render_html_report(report)

    def generate_technical_report(self):
        """Generate technical processing report"""
        return {
            "performance_metrics": self.get_performance_metrics(),
            "operation_analysis": self.analyze_operations(),
            "optimization_suggestions": self.generate_optimization_suggestions(),
            "system_info": self.get_system_info()
        }

    def render_html_report(self, report_data):
        """Render report as HTML document"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .comparison {{ display: flex; gap: 20px; }}
                .image-container {{ text-align: center; }}
                .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
                .metric {{ background: #f5f5f5; padding: 15px; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>Generated on {timestamp}</p>

            <h2>Image Comparison</h2>
            <div class="comparison">
                <div class="image-container">
                    <h3>Original</h3>
                    <img src="{original_image}" alt="Original Image">
                </div>
                <div class="image-container">
                    <h3>Processed</h3>
                    <img src="{processed_image}" alt="Processed Image">
                </div>
            </div>

            <h2>Processing Metrics</h2>
            <div class="metrics">
                <div class="metric">
                    <h4>Total Time</h4>
                    <p>{total_time}s</p>
                </div>
                <div class="metric">
                    <h4>Memory Usage</h4>
                    <p>{memory_usage} MB</p>
                </div>
                <div class="metric">
                    <h4>Cache Hits</h4>
                    <p>{cache_hits}%</p>
                </div>
            </div>

            <h2>Pipeline Steps</h2>
            {pipeline_steps_html}
        </body>
        </html>
        """

        return html_template.format(**report_data)
```

### URL Sharing System
```python
def generate_shareable_url():
    """Generate URL with encoded pipeline configuration"""
    config = get_pipeline_config()
    encoded_config = encode_config_to_url_params(config)

    base_url = st.get_option("browser.serverAddress") or "http://localhost:8501"
    shareable_url = f"{base_url}?config={encoded_config}"

    return shareable_url

def encode_config_to_url_params(config):
    """Encode pipeline configuration for URL sharing"""
    # Compress and base64 encode the configuration
    config_json = json.dumps(config)
    compressed = gzip.compress(config_json.encode('utf-8'))
    encoded = base64.urlsafe_b64encode(compressed).decode('utf-8')

    return encoded

def decode_config_from_url_params(encoded_config):
    """Decode pipeline configuration from URL"""
    try:
        compressed = base64.urlsafe_b64decode(encoded_config.encode('utf-8'))
        config_json = gzip.decompress(compressed).decode('utf-8')
        return json.loads(config_json)
    except Exception as e:
        st.error(f"Failed to decode shared configuration: {e}")
        return None

def load_shared_configuration():
    """Load configuration from URL parameters"""
    query_params = st.experimental_get_query_params()

    if "config" in query_params:
        encoded_config = query_params["config"][0]
        config = decode_config_from_url_params(encoded_config)

        if config:
            load_pipeline_from_config(config)
            st.success("Shared pipeline configuration loaded!")
            return True

    return False
```

### Batch Export System
```python
def render_batch_export_options():
    """Batch export interface for multiple results"""
    st.subheader("Batch Export")

    if not st.session_state.execution_results:
        st.info("No pipeline results available for batch export")
        return

    # Export options
    export_steps = st.multiselect(
        "Select steps to export",
        options=range(len(st.session_state.execution_results)),
        default=list(range(len(st.session_state.execution_results))),
        format_func=lambda x: f"Step {x+1}: {st.session_state.execution_results[x]['operation']}"
    )

    # Format options
    batch_format = st.selectbox("Export format", ["PNG", "JPEG", "WebP"])

    # Naming pattern
    naming_pattern = st.text_input(
        "Naming pattern",
        value="step_{step:02d}_{operation}",
        help="Use {step}, {operation}, {timestamp} as placeholders"
    )

    if st.button("Export All Selected Steps"):
        export_batch_results(export_steps, batch_format, naming_pattern)

def export_batch_results(steps, format_type, naming_pattern):
    """Export multiple pipeline step results"""
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for step_index in steps:
            result = st.session_state.execution_results[step_index]
            image = result['image']

            # Generate filename
            filename = naming_pattern.format(
                step=step_index + 1,
                operation=result['operation'].lower(),
                timestamp=int(time.time())
            ) + f".{format_type.lower()}"

            # Convert image to bytes
            img_buffer = BytesIO()
            image.save(img_buffer, format=format_type, quality=95)
            img_bytes = img_buffer.getvalue()

            # Add to zip
            zip_file.writestr(filename, img_bytes)

        # Add pipeline configuration
        config_json = export_pipeline_config("json")
        zip_file.writestr("pipeline_config.json", config_json)

        # Add processing report
        report_html = generate_processing_report()
        zip_file.writestr("processing_report.html", report_html)

    # Download zip file
    st.download_button(
        label="Download Batch Export",
        data=zip_buffer.getvalue(),
        file_name=f"pipeline_results_{int(time.time())}.zip",
        mime="application/zip"
    )
```

## Advanced Features

### Preset Management System
- Save current pipeline as named preset
- Load presets from personal library
- Share presets with community
- Import/export preset collections

### Version Control
- Track pipeline modifications
- Compare different versions
- Rollback to previous versions
- Merge pipeline configurations

### Performance Optimization
- Lazy loading for large exports
- Background processing for batch exports
- Progress indicators for long operations
- Memory-efficient image handling

## Integration Points

### With Image Display (Task 020)
- Export comparison images
- Include visualization in reports
- Generate step-by-step documentation

### With Pipeline Configuration (Task 018)
- Export parameter configurations
- Import shared parameter sets
- Validate imported configurations

### With Real-time Execution (Task 019)
- Export performance metrics
- Include execution statistics in reports
- Share optimization recommendations

## Success Criteria
- [ ] Images export correctly in multiple formats and sizes
- [ ] Pipeline configurations save and load successfully
- [ ] Processing reports provide comprehensive documentation
- [ ] URL sharing enables easy pipeline sharing
- [ ] Batch export handles multiple results efficiently
- [ ] Generated Python code reproduces pipeline results
- [ ] Export operations are performant and user-friendly
- [ ] Shared configurations load correctly in new sessions

## Test Cases to Implement
- **Image Export Test**: Various formats and sizes export correctly
- **Configuration Export Test**: Pipeline configs save/load without data loss
- **URL Sharing Test**: Shared URLs reproduce original pipeline
- **Report Generation Test**: Reports contain accurate information
- **Batch Export Test**: Multiple results export in organized format
- **Python Code Test**: Generated code produces equivalent results
- **Performance Test**: Export operations complete in reasonable time

## Dependencies
- Builds on: Task 020 (Real-time Image Display and Comparison)
- Required: File download capabilities, URL parameter handling
- Completes: Full Streamlit UI system

## Notes
- Focus on ease of sharing and reproducibility
- Ensure exported configurations are portable
- Make reports useful for both technical and non-technical users
- Optimize for common sharing workflows
- Consider long-term compatibility of exported formats
