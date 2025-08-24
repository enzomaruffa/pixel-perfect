"""Operation browser component for selecting and adding operations to pipeline."""

import streamlit as st

from .operation_registry import OPERATION_REGISTRY, get_operation_by_name, search_operations


def render_operation_browser():
    """Render the operation browser in the sidebar."""

    st.header("🛠️ Operations")

    # Search functionality
    search_query = st.text_input(
        "Search operations...",
        placeholder="Filter by name, description, or tags",
        key="operation_search",
    )

    # Filter controls
    with st.expander("🔍 Filters", expanded=False):
        # Category filter
        all_categories = list(OPERATION_REGISTRY.keys())

        selected_categories = st.multiselect(
            "Categories",
            options=all_categories,
            format_func=lambda x: f"{OPERATION_REGISTRY[x]['icon']} {OPERATION_REGISTRY[x]['name']}",
            key="category_filter",
        )

        # Difficulty filter
        difficulty_filter = st.selectbox(
            "Difficulty Level",
            options=[None, "Beginner", "Intermediate", "Advanced"],
            format_func=lambda x: "All Levels" if x is None else x,
            key="difficulty_filter",
        )

    # Show search results if search query exists
    if search_query:
        st.subheader("🔍 Search Results")
        results = search_operations(
            search_query,
            categories=selected_categories if selected_categories else None,
            difficulty=difficulty_filter,
        )

        if results:
            for op_info in results:
                render_operation_card(op_info, show_category=True)
        else:
            st.info("No operations found matching your search criteria.")

    # Show categorized operations
    else:
        categories_to_show = (
            selected_categories if selected_categories else OPERATION_REGISTRY.keys()
        )

        for category_key in categories_to_show:
            category_info = OPERATION_REGISTRY[category_key]

            with st.expander(f"{category_info['icon']} {category_info['name']}", expanded=True):
                st.caption(category_info["description"])

                # Filter operations by difficulty if specified
                operations = category_info["operations"]
                if difficulty_filter:
                    operations = {
                        name: info
                        for name, info in operations.items()
                        if info["difficulty"] == difficulty_filter
                    }

                if not operations:
                    difficulty_text = difficulty_filter.lower() if difficulty_filter else "matching"
                    st.info(f"No {difficulty_text} operations in this category.")
                    continue

                for op_name, op_info in operations.items():
                    operation_data = {
                        "name": op_name,
                        "category": category_key,
                        "category_name": category_info["name"],
                        "category_icon": category_info["icon"],
                        **op_info,
                    }
                    render_operation_card(operation_data, show_category=False)


def render_operation_card(op_info: dict, show_category: bool = False):
    """Render an individual operation card."""

    # Difficulty color mapping
    difficulty_colors = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴"}

    difficulty_icon = difficulty_colors.get(op_info["difficulty"], "⚪")

    # Operation header
    col1, col2 = st.columns([3, 1])

    with col1:
        if show_category:
            st.write(f"**{op_info['name']}** {difficulty_icon}")
            st.caption(f"{op_info['category_icon']} {op_info['category_name']}")
        else:
            st.write(f"**{op_info['name']}** {difficulty_icon}")

    with col2:
        # Add to pipeline button
        if st.button(
            "➕ Add",
            key=f"add_{op_info['name']}",
            help=f"Add {op_info['name']} to pipeline",
            use_container_width=True,
        ):
            add_operation_to_pipeline(op_info)

    # Operation description
    st.write(op_info["description"])

    # Tags
    if op_info.get("tags"):
        tag_text = " • ".join([f"`{tag}`" for tag in op_info["tags"][:3]])  # Show first 3 tags
        st.caption(tag_text)

    # Info button for detailed view
    if st.button("ℹ️ Details", key=f"info_{op_info['name']}", help="Show detailed information"):
        show_operation_details(op_info)

    st.divider()


def add_operation_to_pipeline(op_info: dict):
    """Add an operation to the pipeline with default parameters."""

    # Initialize pipeline operations if not exists
    if "pipeline_operations" not in st.session_state:
        st.session_state.pipeline_operations = []

    # Create operation configuration
    operation_config = {
        "name": op_info["name"],
        "class": op_info["class"],
        "params": op_info["default_params"].copy(),
        "enabled": True,
        "id": f"{op_info['name']}_{len(st.session_state.pipeline_operations)}",  # Unique ID
    }

    # Add to pipeline
    st.session_state.pipeline_operations.append(operation_config)

    # Trigger auto-execution
    st.session_state.parameters_changed = True

    # Show success message
    st.success(f"✅ Added {op_info['name']} to pipeline!")

    # Trigger rerun to update UI
    st.rerun()


def show_operation_details(op_info: dict):
    """Show detailed operation information in a modal."""

    # Use session state to track which operation details to show
    st.session_state.show_operation_details = op_info["name"]

    # The modal will be rendered in the main area by checking this session state
    st.rerun()


def render_operation_details_modal():
    """Render operation details modal if requested."""

    if not st.session_state.get("show_operation_details"):
        return

    operation_name = st.session_state.show_operation_details
    op_info = get_operation_by_name(operation_name)

    if not op_info:
        st.session_state.show_operation_details = None
        return

    # Modal container
    with st.container():
        st.header(f"ℹ️ {operation_name} Details")

        # Close button
        if st.button("✖️ Close", key="close_details"):
            st.session_state.show_operation_details = None
            st.rerun()

        # Operation information
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Overview")
            st.write(f"**Difficulty:** {op_info['difficulty']}")

            if op_info.get("tags"):
                st.write("**Tags:**")
                for tag in op_info["tags"]:
                    st.write(f"• `{tag}`")

        with col2:
            st.subheader("Description")
            st.write(op_info["description"])

            st.subheader("Default Parameters")
            for param_name, param_value in op_info["default_params"].items():
                st.code(f"{param_name}: {param_value}")

        # Add to pipeline button
        st.divider()
        if st.button(f"➕ Add {operation_name} to Pipeline", type="primary"):
            add_operation_to_pipeline(
                {
                    "name": operation_name,
                    "class": op_info["class"],
                    "default_params": op_info["default_params"],
                    "tags": op_info.get("tags", []),
                }
            )
            st.session_state.show_operation_details = None
            st.rerun()


def render_pipeline_summary():
    """Render a summary of operations in the current pipeline."""

    # Use the enhanced pipeline renderer from parameter_forms
    from ui.components.parameter_forms import render_pipeline_with_edit_buttons

    render_pipeline_with_edit_buttons()
