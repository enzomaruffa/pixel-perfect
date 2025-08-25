# Task 024: Fix Preview Update System

**Status:** Todo
**Priority:** Critical (Week 1)
**Category:** Core Functionality

## Problem

Preview doesn't update reliably after pipeline execution. Users often need to run operations twice to see results in the sidebar preview.

## Root Cause

Streamlit's image caching + session state update timing issues:
- `st.image()` caches results based on input parameters
- Session state updates don't trigger immediate UI refresh
- Preview component doesn't invalidate cache when processed image changes

## Solution

### 1. Force UI Refresh
- Add `st.rerun()` after successful pipeline execution
- Ensure session state is properly synchronized before rerun

### 2. Unique Cache Keys
- Generate unique keys for preview images based on operation hash
- Add timestamp or execution ID to image keys to prevent caching
- Use different image objects to force re-render

### 3. Session State Sync
- Explicit `st.session_state.processed_image` updates
- Clear cache indicators when operations change
- Proper state management in execution flow

## Implementation

### Files to Modify:
- `src/ui/components/layout.py` - `execute_pipeline_inline()` function
- `src/ui/components/layout.py` - `render_sidebar_preview()` function
- `src/ui/components/pipeline_executor.py` - execution flow

### Key Changes:
1. Add `st.rerun()` immediately after setting processed_image
2. Use unique keys for st.image components
3. Add execution timestamps to prevent caching
4. Ensure proper session state updates

## Acceptance Criteria

- [ ] Preview updates immediately after "Execute Now" button
- [ ] No need to run operations twice
- [ ] Sidebar preview always shows latest result
- [ ] No stale image caching issues
- [ ] Works consistently across all operations

## Testing

- Add multiple operations and execute pipeline
- Verify preview updates immediately
- Test with different image sizes and formats
- Ensure no performance degradation from frequent reruns
