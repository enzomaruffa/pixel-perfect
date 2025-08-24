"""Session state management for the Streamlit application."""

import streamlit as st


def initialize_session_state():
    """Initialize all session state variables with default values."""

    # Image state
    if "original_image" not in st.session_state:
        st.session_state.original_image = None

    if "processed_image" not in st.session_state:
        st.session_state.processed_image = None

    # Pipeline state
    if "pipeline_operations" not in st.session_state:
        st.session_state.pipeline_operations = []

    # Execution state
    if "execution_results" not in st.session_state:
        st.session_state.execution_results = []

    if "execution_in_progress" not in st.session_state:
        st.session_state.execution_in_progress = False

    if "execution_error" not in st.session_state:
        st.session_state.execution_error = None

    # UI state
    if "selected_operation" not in st.session_state:
        st.session_state.selected_operation = None

    if "show_advanced_options" not in st.session_state:
        st.session_state.show_advanced_options = False


def reset_pipeline():
    """Reset the pipeline to initial state."""
    st.session_state.pipeline_operations = []
    st.session_state.execution_results = []
    st.session_state.processed_image = None
    st.session_state.execution_error = None
    st.session_state.selected_operation = None


def reset_all():
    """Reset entire application state."""
    st.session_state.original_image = None
    st.session_state.processed_image = None
    reset_pipeline()


def get_pipeline_summary() -> dict:
    """Get a summary of the current pipeline state."""
    return {
        "has_image": st.session_state.original_image is not None,
        "operation_count": len(st.session_state.pipeline_operations),
        "has_results": len(st.session_state.execution_results) > 0,
        "is_processing": st.session_state.execution_in_progress,
        "has_error": st.session_state.execution_error is not None,
    }
