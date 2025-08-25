# Task 032: Visual Design Cleanup

**Status:** Todo
**Priority:** Medium (Week 3)
**Category:** Polish & Performance

## Problem

Inconsistent visual design throughout the app:
- No unified color scheme or typography
- Inconsistent spacing and layout
- Mixed icon styles and sizes
- Amateur appearance that doesn't inspire confidence

## Goal

Create a clean, professional, consistent visual design that screams "power user tool" - functional, efficient, polished.

## Solution

### 1. Design System
Establish consistent visual language:

**Color Palette:**
- Primary: Clean blues (#0066CC, #E6F2FF)
- Success: Green (#10B981, #D1FAE5)
- Warning: Amber (#F59E0B, #FEF3C7)
- Error: Red (#DC2626, #FEE2E2)
- Neutral: Grays (#6B7280, #F9FAFB, #374151)

**Typography:**
- Headers: Bold, clear hierarchy
- Body: Readable sans-serif
- Code/Formulas: Monospace font

**Spacing:**
- Consistent 8px grid system
- Proper whitespace between sections
- Aligned elements and margins

### 2. Icon System
- Consistent emoji-based icons throughout
- Meaningful associations (⚙️ settings, 🔧 tools, 📁 files)
- Proper size and spacing
- No mixing of different icon styles

### 3. Layout Improvements
- Clear visual hierarchy
- Logical content grouping
- Better use of whitespace
- Consistent button styles and sizes
- Proper alignment of all elements

## Implementation

### Files to Modify:
- `src/ui/components/layout.py` - main layout styling
- `src/ui/components/operation_browser.py` - operation display
- `src/ui/components/parameter_forms.py` - form styling
- `src/ui/components/export_manager.py` - export UI

### Key Changes:

1. **Consistent Button Styling:**
   ```python
   # Standardize all buttons
   def create_primary_button(label, key, help=None):
       return st.button(label, type="primary", help=help, key=key)

   def create_secondary_button(label, key, help=None):
       return st.button(label, help=help, key=key)
   ```

2. **Unified Header System:**
   ```python
   def render_section_header(title, icon="", description=None):
       if icon:
           st.subheader(f"{icon} {title}")
       else:
           st.subheader(title)

       if description:
           st.caption(description)

       st.divider()  # Consistent section separation
   ```

3. **Standardized Spacing:**
   ```python
   # Use consistent spacing helpers
   def add_vertical_space(size="medium"):
       sizes = {"small": 1, "medium": 2, "large": 4}
       for _ in range(sizes.get(size, 2)):
           st.write("")
   ```

4. **Clean Operation Display:**
   ```python
   def render_operation_card(op_info):
       with st.container():
           st.markdown(f"### {op_info['icon']} {op_info['name']}")
           st.markdown(f"*{op_info['description']}*")

           # Consistent button styling
           if st.button("Add to Pipeline", key=f"add_{op_info['name']}",
                       type="primary", use_container_width=True):
               add_operation_to_pipeline(op_info)
   ```

### 5. Professional Color Usage:
```python
# Use Streamlit's built-in styling consistently
def show_status(message, status="info"):
    if status == "success":
        st.success(message)
    elif status == "error":
        st.error(message)
    elif status == "warning":
        st.warning(message)
    else:
        st.info(message)
```

## Visual Standards

### Headers:
- Main title: `st.title("🎨 Pixel Perfect")`
- Section headers: `st.subheader("⚙️ Operation Settings")`
- Subsections: `st.write("**Parameter Configuration**")`

### Status Indicators:
- Success: ✅ Green checkmark
- Error: ❌ Red X
- Warning: ⚠️ Yellow triangle
- Info: ℹ️ Blue info
- Processing: ⏳ Hourglass

### Button Hierarchy:
- Primary actions: `type="primary"` (blue)
- Secondary actions: default (white)
- Destructive actions: red warning text

### Layout Patterns:
- Two-column layouts for related content
- Three-column layouts for action groups
- Consistent column proportions
- Proper spacing between sections

## Acceptance Criteria

- [ ] Consistent color usage throughout app
- [ ] Unified typography and spacing
- [ ] Professional button and form styling
- [ ] Clean, organized layout hierarchy
- [ ] Consistent icon usage and sizing
- [ ] No visual clutter or inconsistencies
- [ ] Polished, professional appearance

## Testing

- Review every page for visual consistency
- Check all buttons use consistent styling
- Verify proper spacing and alignment
- Ensure readable typography hierarchy
- Test responsive layout behavior
