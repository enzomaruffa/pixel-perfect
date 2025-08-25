"""Pipeline save/load management UI components."""

import streamlit as st
import uuid
from datetime import datetime
from typing import Dict, Any, List

from ui.utils.pipeline_serialization import (
    serialize_pipeline,
    deserialize_pipeline, 
    export_pipeline_json,
    import_pipeline_json,
    validate_pipeline_format
)


def render_pipeline_save_load():
    """Render pipeline save/load interface."""
    st.subheader("💾 Pipeline Management")
    
    # Check if there's a pipeline to save
    has_pipeline = bool(st.session_state.get("pipeline_operations", []))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Save Pipeline**")
        render_save_pipeline(has_pipeline)
    
    with col2:
        st.write("**Load Pipeline**")
        render_load_pipeline()


def render_save_pipeline(has_pipeline: bool):
    """Render pipeline save interface."""
    
    if not has_pipeline:
        st.info("Add operations to your pipeline to save it")
        return
    
    # Pipeline save form
    with st.form("save_pipeline_form", clear_on_submit=False):
        pipeline_name = st.text_input(
            "Pipeline Name",
            value="My Pipeline",
            help="Name for your saved pipeline"
        )
        
        pipeline_description = st.text_area(
            "Description (Optional)",
            value="",
            help="Brief description of what this pipeline does"
        )
        
        save_clicked = st.form_submit_button("💾 Save Pipeline", type="primary")
        
        if save_clicked:
            try:
                # Serialize current pipeline
                operations = st.session_state.get("pipeline_operations", [])
                pipeline_data = serialize_pipeline(
                    operations,
                    name=pipeline_name.strip() or "Untitled Pipeline",
                    description=pipeline_description.strip()
                )
                
                # Convert to JSON
                json_str = export_pipeline_json(pipeline_data)
                
                # Generate filename
                safe_name = "".join(c for c in pipeline_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                filename = f"{safe_name or 'pipeline'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                # Offer download
                st.download_button(
                    label="📥 Download Pipeline",
                    data=json_str,
                    file_name=filename,
                    mime="application/json",
                    help=f"Download {pipeline_name} as JSON file"
                )
                
                st.success(f"✅ Pipeline '{pipeline_name}' ready for download!")
                
                # Show pipeline summary
                with st.expander("📋 Pipeline Summary", expanded=False):
                    st.write(f"**Operations:** {len(operations)}")
                    for i, op in enumerate(operations, 1):
                        status = "✅" if op.get("enabled", True) else "❌"
                        st.write(f"{i}. {status} {op['name']}")
                
            except Exception as e:
                st.error(f"❌ Failed to save pipeline: {str(e)}")


def render_load_pipeline():
    """Render pipeline load interface."""
    
    uploaded_file = st.file_uploader(
        "Upload Pipeline",
        type=['json'],
        help="Upload a previously saved pipeline JSON file"
    )
    
    if uploaded_file is not None:
        try:
            # Read and parse JSON
            json_content = uploaded_file.read().decode('utf-8')
            pipeline_data = import_pipeline_json(json_content)
            
            # Validate format
            if not validate_pipeline_format(pipeline_data):
                st.error("❌ Invalid pipeline format")
                return
            
            # Show pipeline info
            pipeline_name = pipeline_data.get("name", "Imported Pipeline")
            pipeline_desc = pipeline_data.get("description", "")
            created_date = pipeline_data.get("created", "Unknown")
            operations_count = len(pipeline_data.get("operations", []))
            
            st.success(f"✅ Found valid pipeline: **{pipeline_name}**")
            
            # Pipeline preview
            with st.expander("👀 Pipeline Preview", expanded=True):
                st.write(f"**Name:** {pipeline_name}")
                if pipeline_desc:
                    st.write(f"**Description:** {pipeline_desc}")
                st.write(f"**Created:** {created_date}")
                st.write(f"**Operations:** {operations_count}")
                
                # List operations
                if operations_count > 0:
                    st.write("**Operation List:**")
                    for i, op in enumerate(pipeline_data["operations"], 1):
                        enabled = op.get("enabled", True)
                        status = "✅" if enabled else "❌"
                        op_name = op.get("name", op.get("type", "Unknown"))
                        st.write(f"{i}. {status} {op_name}")
            
            # Load options
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Replace Current Pipeline", type="primary"):
                    load_pipeline_to_session(pipeline_data, replace=True)
                    st.success(f"✅ Replaced pipeline with '{pipeline_name}'")
                    st.rerun()
            
            with col2:
                if st.button("➕ Add to Current Pipeline"):
                    load_pipeline_to_session(pipeline_data, replace=False)
                    st.success(f"✅ Added operations from '{pipeline_name}'")
                    st.rerun()
                    
        except Exception as e:
            st.error(f"❌ Failed to load pipeline: {str(e)}")
            st.caption("Make sure the file is a valid pipeline JSON file")


def load_pipeline_to_session(pipeline_data: Dict[str, Any], replace: bool = True):
    """Load pipeline data into session state.
    
    Args:
        pipeline_data: Deserialized pipeline data
        replace: If True, replace current pipeline. If False, append operations.
    """
    try:
        # Deserialize operations
        operations = deserialize_pipeline(pipeline_data)
        
        if replace:
            # Replace entire pipeline
            st.session_state.pipeline_operations = operations
        else:
            # Append to existing pipeline
            if "pipeline_operations" not in st.session_state:
                st.session_state.pipeline_operations = []
            
            # Generate new IDs for imported operations to avoid conflicts
            for op in operations:
                op["id"] = str(uuid.uuid4())
            
            st.session_state.pipeline_operations.extend(operations)
        
        # Mark parameters as changed for auto-execution
        st.session_state.parameters_changed = True
        
    except Exception as e:
        raise Exception(f"Failed to load pipeline: {str(e)}")


def render_quick_save_button():
    """Render a quick save button for the main interface."""
    operations = st.session_state.get("pipeline_operations", [])
    
    if not operations:
        return
    
    if st.button("💾 Save", help="Save current pipeline"):
        # Generate auto-name based on operations
        op_names = [op["name"] for op in operations[:3]]  # First 3 operations
        auto_name = " + ".join(op_names)
        if len(operations) > 3:
            auto_name += f" + {len(operations) - 3} more"
        
        # Quick save with auto-generated name
        pipeline_data = serialize_pipeline(
            operations,
            name=auto_name,
            description=f"Auto-saved pipeline with {len(operations)} operations"
        )
        
        json_str = export_pipeline_json(pipeline_data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"pipeline_{timestamp}.json"
        
        st.download_button(
            label="📥 Download",
            data=json_str,
            file_name=filename,
            mime="application/json",
            help="Download pipeline file"
        )


def render_pipeline_actions():
    """Render pipeline action buttons for integration in main UI."""
    operations = st.session_state.get("pipeline_operations", [])
    
    if not operations:
        st.info("Add operations to enable pipeline management")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        render_quick_save_button()
    
    with col2:
        if st.button("🔄 Load", help="Load saved pipeline"):
            # Set flag to show load interface
            st.session_state.show_load_interface = True
            st.rerun()
    
    with col3:
        if st.button("📋 Manage", help="Full pipeline management"):
            # Set flag to show full management interface
            st.session_state.show_pipeline_manager = True
            st.rerun()