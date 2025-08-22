"""Interactive pipeline builder with live preview - k9s style interface."""

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from PIL import Image
from rich.console import Console
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Button, DataTable, Footer, Header, Input, Label, ListItem, ListView, 
    OptionList, Rule, Static, TextArea, Tree
)
from textual.widgets.option_list import Option

from core.context import ImageContext
from core.pipeline import Pipeline
from operations import *
from presets.built_in import get_all_presets
from utils.cache_manager import CachePolicy

# Import all operations for dynamic loading
OPERATION_CLASSES = {
    "PixelFilter": PixelFilter, "PixelMath": PixelMath, "PixelSort": PixelSort,
    "RowShift": RowShift, "RowStretch": RowStretch, "RowRemove": RowRemove, "RowShuffle": RowShuffle,
    "ColumnShift": ColumnShift, "ColumnStretch": ColumnStretch, "ColumnMirror": ColumnMirror, "ColumnWeave": ColumnWeave,
    "BlockFilter": BlockFilter, "BlockShift": BlockShift, "BlockRotate": BlockRotate, "BlockScramble": BlockScramble,
    "GridWarp": GridWarp, "PerspectiveStretch": PerspectiveStretch, "RadialStretch": RadialStretch,
    "AspectStretch": AspectStretch, "AspectCrop": AspectCrop, "AspectPad": AspectPad,
    "ChannelSwap": ChannelSwap, "ChannelIsolate": ChannelIsolate, "AlphaGenerator": AlphaGenerator,
    "Mosaic": Mosaic, "Dither": Dither,
}

OPERATION_CATEGORIES = {
    "🎯 Pixel": ["PixelFilter", "PixelMath", "PixelSort"],
    "↔️ Row": ["RowShift", "RowStretch", "RowRemove", "RowShuffle"],
    "↕️ Column": ["ColumnShift", "ColumnStretch", "ColumnMirror", "ColumnWeave"],
    "🔳 Block": ["BlockFilter", "BlockShift", "BlockRotate", "BlockScramble"],
    "🌊 Geometric": ["GridWarp", "PerspectiveStretch", "RadialStretch"],
    "📐 Aspect": ["AspectStretch", "AspectCrop", "AspectPad"],
    "🎨 Channel": ["ChannelSwap", "ChannelIsolate", "AlphaGenerator"],
    "🎭 Pattern": ["Mosaic", "Dither"],
}


class ImagePreview(Static):
    """Widget to display ASCII art preview of image."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_path: Optional[Path] = None
        self.preview_text = "No image loaded"
        self.update(self.preview_text)
        
    def set_image(self, image_path: Path) -> None:
        """Set the image to preview."""
        self.image_path = image_path
        self.update_preview()
    
    def update_preview(self) -> None:
        """Update the ASCII preview."""
        if not self.image_path or not self.image_path.exists():
            self.preview_text = "❌ Image not found"
            self.update(self.preview_text)
            return
            
        try:
            # Load and resize image for ASCII preview
            image = Image.open(self.image_path)
            original_size = image.size
            
            # Resize to fit terminal (roughly)
            target_width = 40
            target_height = 20
            aspect_ratio = image.width / image.height
            
            if aspect_ratio > 1:
                width = target_width
                height = int(target_width / aspect_ratio)
            else:
                height = target_height
                width = int(target_height * aspect_ratio)
            
            image = image.resize((width, height))
            
            # Convert to grayscale for ASCII
            if image.mode != 'L':
                image = image.convert('L')
            
            # ASCII characters from light to dark (inverted for proper display)
            ascii_chars = " .:-=+*#%@"
            
            # Generate ASCII art
            ascii_lines = []
            for y in range(image.height):
                line = ""
                for x in range(image.width):
                    pixel = image.getpixel((x, y))
                    char_index = min(len(ascii_chars) - 1, pixel * len(ascii_chars) // 256)
                    line += ascii_chars[char_index]
                ascii_lines.append(line)
            
            # Add image info
            info = f"📐 {original_size[0]}×{original_size[1]} | 📊 {image.mode}"
            self.preview_text = f"{info}\n\n" + "\n".join(ascii_lines)
            
            self.update(self.preview_text)
            
        except Exception as e:
            self.preview_text = f"❌ Preview error: {e}"
            self.update(self.preview_text)
    
    def render(self) -> Text:
        return Text(self.preview_text, style="cyan")


class OperationBrowser(Container):
    """Widget for browsing and selecting operations."""
    
    def compose(self) -> ComposeResult:
        yield Label("🔍 Operation Browser", classes="title")
        yield Rule()
        
        # Search input
        yield Input(placeholder="Search operations...", id="search_input")
        
        # Category tree
        tree = Tree("Operations", id="op_tree")
        tree.show_root = False
        
        for category, operations in OPERATION_CATEGORIES.items():
            category_node = tree.root.add(category, expand=True)
            for op_name in operations:
                op_class = OPERATION_CLASSES[op_name]
                doc = op_class.__doc__ or "No description"
                short_desc = doc.split('\n')[0].strip()
                category_node.add_leaf(f"{op_name}: {short_desc[:30]}...")
        
        yield tree
        
        # Presets section
        yield Rule()
        yield Label("🎭 Presets", classes="subtitle")
        
        # Create preset list items
        preset_items = []
        presets = get_all_presets()
        for preset_name, preset_data in presets.items():
            description = preset_data.get("description", "No description")
            preset_items.append(ListItem(Label(f"{preset_name}: {description[:40]}...")))
        
        # Yield the ListView with items
        yield ListView(*preset_items, id="preset_list")


class PipelineEditor(Container):
    """Widget for editing the processing pipeline."""
    
    pipeline_steps = reactive([])
    current_step = reactive(0)
    
    def compose(self) -> ComposeResult:
        yield Label("🔧 Pipeline Editor", classes="title")
        yield Rule()
        
        # Pipeline steps table
        table = DataTable(id="pipeline_table")
        table.add_columns("Step", "Operation", "Status", "Time", "Cache")
        yield table
        
        # Step controls
        with Horizontal():
            yield Button("▲ Move Up", id="move_up", variant="primary")
            yield Button("▼ Move Down", id="move_down", variant="primary")
            yield Button("❌ Remove", id="remove_step", variant="error")
            yield Button("⚙️ Edit", id="edit_step", variant="success")
        
        yield Rule()
        yield Label("📊 Performance", classes="subtitle")
        yield Static("No pipeline executed yet", id="performance_stats")


class ParameterEditor(Container):
    """Widget for editing operation parameters."""
    
    current_operation = reactive(None)
    
    def compose(self) -> ComposeResult:
        yield Label("⚙️ Parameter Editor", classes="title")
        yield Rule()
        yield Static("Select an operation to edit parameters", id="param_content")
        
        # Save/Cancel buttons
        with Horizontal():
            yield Button("💾 Save", id="save_params", variant="success")
            yield Button("❌ Cancel", id="cancel_params", variant="error")


class LivePreview(Container):
    """Widget for live image preview with step-by-step execution."""
    
    current_step = reactive(-1)  # -1 means show original
    
    def compose(self) -> ComposeResult:
        yield Label("🖼️ Live Preview", classes="title")
        yield Rule()
        
        # Step controls
        with Horizontal():
            yield Button("⏮️ Original", id="preview_original", variant="primary")
            yield Button("⏪ Previous", id="preview_prev", variant="primary") 
            yield Button("⏩ Next", id="preview_next", variant="primary")
            yield Button("⏭️ Final", id="preview_final", variant="primary")
        
        yield Rule()
        yield Label("Step: Original Image", id="step_label")
        yield ImagePreview(id="image_preview")
        
        yield Rule()
        yield Label("📈 Step Info", classes="subtitle")
        yield Static("Original image loaded", id="step_info")


class InteractivePipelineApp(App):
    """Main interactive pipeline builder application."""
    
    CSS = """
    .title {
        text-style: bold;
        color: cyan;
    }
    
    .subtitle {
        text-style: bold;
        color: yellow;
    }
    
    #left_panel {
        width: 1fr;
    }
    
    #right_panel {
        width: 1fr;
    }
    
    #operation_browser {
        height: 1fr;
    }
    
    #pipeline_editor {
        height: 1fr;
    }
    
    #parameter_editor {
        height: 40%;
    }
    
    #live_preview {
        height: 1fr;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+s", "save_pipeline", "Save Pipeline"),
        Binding("ctrl+o", "load_pipeline", "Load Pipeline"),
        Binding("ctrl+r", "run_pipeline", "Run Pipeline"),
        Binding("ctrl+e", "export_config", "Export Config"),
        Binding("space", "toggle_step", "Toggle Step"),
        Binding("enter", "add_operation", "Add Operation"),
        Binding("delete", "remove_step", "Remove Step"),
        Binding("f1", "help", "Help"),
    ]
    
    def __init__(self, input_path: Path, output_path: Path, cache_dir: Optional[Path] = None):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.cache_dir = cache_dir
        self.pipeline = Pipeline(input_path, debug=True, cache_dir=cache_dir)
        self.pipeline_steps: List[Dict[str, Any]] = []
        self.current_step = -1
        self.temp_images: List[Path] = []  # Store intermediate results
        
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Horizontal():
            # Left panel: Operation browser and pipeline editor
            with Vertical(id="left_panel"):
                yield OperationBrowser(id="operation_browser")
                yield PipelineEditor(id="pipeline_editor")
            
            # Right panel: Parameter editor and live preview
            with Vertical(id="right_panel"):
                yield ParameterEditor(id="parameter_editor")
                yield LivePreview(id="live_preview")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the app when mounted."""
        self.title = "🎨 Pixel Perfect - Interactive Pipeline Builder"
        self.sub_title = f"Input: {self.input_path.name} → Output: {self.output_path.name}"
        
        # Load initial image preview
        preview = self.query_one("#image_preview", ImagePreview)
        preview.set_image(self.input_path)
        
        # Initialize pipeline table
        self.update_pipeline_table()
    
    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle operation selection from tree."""
        # Check if it's a leaf node (has no children)
        if not event.node.children:
            # Extract operation name from the label
            label = str(event.node.label)
            op_name = label.split(":")[0].strip()
            if op_name in OPERATION_CLASSES:
                self.add_operation_to_pipeline(op_name)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "move_up":
            self.move_step_up()
        elif button_id == "move_down":
            self.move_step_down()
        elif button_id == "remove_step":
            self.remove_current_step()
        elif button_id == "edit_step":
            self.edit_current_step()
        elif button_id == "preview_original":
            self.preview_step(-1)
        elif button_id == "preview_prev":
            self.preview_step(max(-1, self.current_step - 1))
        elif button_id == "preview_next":
            self.preview_step(min(len(self.pipeline_steps) - 1, self.current_step + 1))
        elif button_id == "preview_final":
            self.preview_step(len(self.pipeline_steps) - 1)
        elif button_id == "save_params":
            self.save_parameters()
        elif button_id == "cancel_params":
            self.cancel_parameter_edit()
    
    def add_operation_to_pipeline(self, op_name: str) -> None:
        """Add an operation to the pipeline."""
        try:
            # Create operation with default parameters
            op_class = OPERATION_CLASSES[op_name]
            operation = op_class()
            
            # Add to pipeline
            self.pipeline.add(operation)
            
            # Add to our step tracking
            step_info = {
                "operation": operation,
                "name": op_name,
                "status": "Ready",
                "time": 0.0,
                "cache_hit": False,
                "params": operation.model_dump()
            }
            self.pipeline_steps.append(step_info)
            
            # Update UI
            self.update_pipeline_table()
            self.notify(f"✅ Added {op_name} to pipeline")
            
            # Auto-preview if enabled
            self.preview_step(len(self.pipeline_steps) - 1)
            
        except Exception as e:
            self.notify(f"❌ Failed to add {op_name}: {e}", severity="error")
    
    def update_pipeline_table(self) -> None:
        """Update the pipeline steps table."""
        table = self.query_one("#pipeline_table", DataTable)
        table.clear()
        
        for i, step in enumerate(self.pipeline_steps):
            status_icon = "✅" if step["status"] == "Complete" else "⏳" if step["status"] == "Running" else "⏸️"
            cache_icon = "💾" if step["cache_hit"] else "🔄"
            time_str = f"{step['time']:.2f}s" if step['time'] > 0 else "-"
            
            table.add_row(
                str(i + 1),
                step["name"],
                f"{status_icon} {step['status']}",
                time_str,
                cache_icon
            )
    
    def preview_step(self, step_index: int) -> None:
        """Preview the pipeline up to a specific step."""
        self.current_step = step_index
        
        try:
            if step_index == -1:
                # Show original image
                preview = self.query_one("#image_preview", ImagePreview)
                preview.set_image(self.input_path)
                
                step_label = self.query_one("#step_label", Label)
                step_label.update("Step: Original Image")
                
                step_info = self.query_one("#step_info", Static)
                step_info.update("Original image - no processing applied")
                
            else:
                # Execute pipeline up to this step
                self.execute_pipeline_to_step(step_index)
        
        except Exception as e:
            self.notify(f"❌ Preview failed: {e}", severity="error")
    
    def execute_pipeline_to_step(self, target_step: int) -> None:
        """Execute pipeline up to a specific step."""
        if target_step >= len(self.pipeline_steps):
            return
        
        try:
            # Create a temporary pipeline with steps up to target
            temp_pipeline = Pipeline(self.input_path, debug=False, cache_dir=self.cache_dir)
            
            for i in range(target_step + 1):
                temp_pipeline.add(self.pipeline_steps[i]["operation"])
            
            # Generate temporary output path
            temp_output = self.output_path.parent / f"temp_step_{target_step}_{self.output_path.name}"
            
            # Execute with timing
            start_time = time.time()
            self.pipeline_steps[target_step]["status"] = "Running"
            self.update_pipeline_table()
            
            temp_pipeline.execute(temp_output)
            
            execution_time = time.time() - start_time
            self.pipeline_steps[target_step]["time"] = execution_time
            self.pipeline_steps[target_step]["status"] = "Complete"
            
            # Update preview
            preview = self.query_one("#image_preview", ImagePreview)
            preview.set_image(temp_output)
            
            # Update step info
            step_label = self.query_one("#step_label", Label)
            step_name = self.pipeline_steps[target_step]["name"]
            step_label.update(f"Step {target_step + 1}: {step_name}")
            
            step_info = self.query_one("#step_info", Static)
            cache_stats = temp_pipeline.get_cache_statistics()
            if cache_stats:
                hit_rate = cache_stats.get("hit_rate", 0) * 100
                info_text = f"Execution time: {execution_time:.2f}s\nCache hit rate: {hit_rate:.1f}%"
            else:
                info_text = f"Execution time: {execution_time:.2f}s\nCache: Not enabled"
            
            step_info.update(info_text)
            
            # Track temp file for cleanup
            self.temp_images.append(temp_output)
            
            # Update table
            self.update_pipeline_table()
            
        except Exception as e:
            self.pipeline_steps[target_step]["status"] = "Error"
            self.update_pipeline_table()
            raise e
    
    def move_step_up(self) -> None:
        """Move current step up in the pipeline."""
        # Implementation for moving steps
        pass
    
    def move_step_down(self) -> None:
        """Move current step down in the pipeline."""
        # Implementation for moving steps
        pass
    
    def remove_current_step(self) -> None:
        """Remove the currently selected step."""
        # Implementation for removing steps
        pass
    
    def edit_current_step(self) -> None:
        """Edit parameters of current step."""
        # Implementation for parameter editing
        pass
    
    def save_parameters(self) -> None:
        """Save edited parameters."""
        pass
    
    def cancel_parameter_edit(self) -> None:
        """Cancel parameter editing."""
        pass
    
    def action_save_pipeline(self) -> None:
        """Save current pipeline to file."""
        config = {
            "operations": [
                {
                    "type": step["name"],
                    "params": step["params"]
                }
                for step in self.pipeline_steps
            ]
        }
        
        config_path = self.output_path.parent / f"{self.output_path.stem}_pipeline.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        self.notify(f"💾 Pipeline saved to {config_path}")
    
    def action_run_pipeline(self) -> None:
        """Execute the full pipeline."""
        if not self.pipeline_steps:
            self.notify("⚠️ No operations in pipeline", severity="warning")
            return
        
        try:
            start_time = time.time()
            self.pipeline.execute(self.output_path)
            total_time = time.time() - start_time
            
            self.notify(f"✅ Pipeline executed in {total_time:.2f}s")
            
            # Update final preview
            preview = self.query_one("#image_preview", ImagePreview)
            preview.set_image(self.output_path)
            
        except Exception as e:
            self.notify(f"❌ Pipeline failed: {e}", severity="error")
    
    def action_help(self) -> None:
        """Show help information."""
        help_text = """
🎨 Pixel Perfect Interactive Pipeline Builder

KEYBOARD SHORTCUTS:
  Ctrl+S    Save pipeline configuration
  Ctrl+O    Load pipeline configuration  
  Ctrl+R    Run full pipeline
  Ctrl+E    Export configuration
  Space     Toggle step execution
  Enter     Add selected operation
  Delete    Remove selected step
  F1        Show this help

USAGE:
1. Browse operations in the left panel
2. Select operations to add to pipeline  
3. Use live preview to see results
4. Edit parameters as needed
5. Save and export your pipeline

Press any key to continue...
        """
        # In a real implementation, this would show a modal dialog
        self.notify("Press F1 for help (see documentation)")
    
    def on_unmount(self) -> None:
        """Cleanup when app is closed."""
        # Clean up temporary files
        for temp_file in self.temp_images:
            if temp_file.exists():
                temp_file.unlink()


def launch_interactive_builder(input_path: Path, output_path: Path, cache_dir: Optional[Path] = None):
    """Launch the interactive pipeline builder."""
    app = InteractivePipelineApp(input_path, output_path, cache_dir)
    app.run()