"""Professional design system for Pixel Perfect UI.

Provides consistent colors, typography, spacing, and component styling
for a polished power-user interface.
"""

import streamlit as st


class Colors:
    """Professional color palette for technical interfaces."""

    # Core brand colors
    PRIMARY_BLUE = "#2563eb"
    PRIMARY_BLUE_LIGHT = "#3b82f6"
    PRIMARY_BLUE_DARK = "#1d4ed8"

    # Success states
    SUCCESS_GREEN = "#059669"
    SUCCESS_GREEN_LIGHT = "#10b981"
    SUCCESS_GREEN_DARK = "#047857"

    # Warning states
    WARNING_AMBER = "#d97706"
    WARNING_AMBER_LIGHT = "#f59e0b"
    WARNING_AMBER_DARK = "#b45309"

    # Error states
    ERROR_RED = "#dc2626"
    ERROR_RED_LIGHT = "#ef4444"
    ERROR_RED_DARK = "#b91c1c"

    # Neutral grays
    GRAY_50 = "#f9fafb"
    GRAY_100 = "#f3f4f6"
    GRAY_200 = "#e5e7eb"
    GRAY_300 = "#d1d5db"
    GRAY_400 = "#9ca3af"
    GRAY_500 = "#6b7280"
    GRAY_600 = "#4b5563"
    GRAY_700 = "#374151"
    GRAY_800 = "#1f2937"
    GRAY_900 = "#111827"

    # Backgrounds (light colors - Streamlit handles dark mode automatically)
    BG_PRIMARY = "#ffffff"
    BG_SECONDARY = "#f8fafc"
    BG_TERTIARY = "#f1f5f9"

    # Text colors (let Streamlit handle dark mode)
    TEXT_PRIMARY = "#1e293b"
    TEXT_SECONDARY = "#475569"
    TEXT_MUTED = "#64748b"

    # Borders (let Streamlit handle dark mode)
    BORDER_LIGHT = "#e2e8f0"
    BORDER_DEFAULT = "#cbd5e1"
    BORDER_STRONG = "#94a3b8"


class Spacing:
    """8px grid spacing system."""

    XS = "4px"  # 0.5 units
    SM = "8px"  # 1 unit
    MD = "16px"  # 2 units
    LG = "24px"  # 3 units
    XL = "32px"  # 4 units
    XXL = "48px"  # 6 units
    XXXL = "64px"  # 8 units


class Typography:
    """Typography scale and font definitions."""

    # Font families
    SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif"
    MONO = "'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace"

    # Font sizes (rem based)
    TEXT_XS = "0.75rem"  # 12px
    TEXT_SM = "0.875rem"  # 14px
    TEXT_BASE = "1rem"  # 16px
    TEXT_LG = "1.125rem"  # 18px
    TEXT_XL = "1.25rem"  # 20px
    TEXT_2XL = "1.5rem"  # 24px
    TEXT_3XL = "1.875rem"  # 30px
    TEXT_4XL = "2.25rem"  # 36px

    # Font weights
    LIGHT = "300"
    NORMAL = "400"
    MEDIUM = "500"
    SEMIBOLD = "600"
    BOLD = "700"


class Components:
    """Styled component helpers."""

    @staticmethod
    def primary_button_style() -> str:
        """CSS for primary action buttons."""
        return f"""
            background-color: {Colors.PRIMARY_BLUE} !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: {Typography.MEDIUM} !important;
            font-size: {Typography.TEXT_SM} !important;
            padding: {Spacing.SM} {Spacing.MD} !important;
            transition: background-color 0.2s ease !important;
        """

    @staticmethod
    def secondary_button_style() -> str:
        """CSS for secondary action buttons."""
        return f"""
            background-color: {Colors.BG_PRIMARY} !important;
            color: {Colors.TEXT_PRIMARY} !important;
            border: 1px solid {Colors.BORDER_DEFAULT} !important;
            border-radius: 6px !important;
            font-weight: {Typography.MEDIUM} !important;
            font-size: {Typography.TEXT_SM} !important;
            padding: {Spacing.SM} {Spacing.MD} !important;
            transition: all 0.2s ease !important;
        """

    @staticmethod
    def success_button_style() -> str:
        """CSS for success/confirmation buttons."""
        return f"""
            background-color: {Colors.SUCCESS_GREEN} !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: {Typography.MEDIUM} !important;
            font-size: {Typography.TEXT_SM} !important;
            padding: {Spacing.SM} {Spacing.MD} !important;
            transition: background-color 0.2s ease !important;
        """

    @staticmethod
    def danger_button_style() -> str:
        """CSS for destructive action buttons."""
        return f"""
            background-color: {Colors.ERROR_RED} !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: {Typography.MEDIUM} !important;
            font-size: {Typography.TEXT_SM} !important;
            padding: {Spacing.SM} {Spacing.MD} !important;
            transition: background-color 0.2s ease !important;
        """

    @staticmethod
    def card_style() -> str:
        """CSS for card containers."""
        return f"""
            background-color: {Colors.BG_PRIMARY};
            border: 1px solid {Colors.BORDER_LIGHT};
            border-radius: 8px;
            padding: {Spacing.LG};
            margin-bottom: {Spacing.MD};
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        """

    @staticmethod
    def section_header_style() -> str:
        """CSS for section headers."""
        return f"""
            font-family: {Typography.SANS};
            font-size: {Typography.TEXT_LG};
            font-weight: {Typography.SEMIBOLD};
            color: {Colors.TEXT_PRIMARY};
            margin-bottom: {Spacing.MD};
            padding-bottom: {Spacing.SM};
            border-bottom: 2px solid {Colors.PRIMARY_BLUE};
        """

    @staticmethod
    def info_badge_style() -> str:
        """CSS for informational badges."""
        return f"""
            background-color: {Colors.PRIMARY_BLUE};
            color: white;
            font-size: {Typography.TEXT_XS};
            font-weight: {Typography.MEDIUM};
            padding: {Spacing.XS} {Spacing.SM};
            border-radius: 12px;
            display: inline-block;
            margin: {Spacing.XS};
        """

    @staticmethod
    def success_badge_style() -> str:
        """CSS for success status badges."""
        return f"""
            background-color: {Colors.SUCCESS_GREEN};
            color: white;
            font-size: {Typography.TEXT_XS};
            font-weight: {Typography.MEDIUM};
            padding: {Spacing.XS} {Spacing.SM};
            border-radius: 12px;
            display: inline-block;
            margin: {Spacing.XS};
        """

    @staticmethod
    def warning_badge_style() -> str:
        """CSS for warning status badges."""
        return f"""
            background-color: {Colors.WARNING_AMBER};
            color: white;
            font-size: {Typography.TEXT_XS};
            font-weight: {Typography.MEDIUM};
            padding: {Spacing.XS} {Spacing.SM};
            border-radius: 12px;
            display: inline-block;
            margin: {Spacing.XS};
        """

    @staticmethod
    def error_badge_style() -> str:
        """CSS for error status badges."""
        return f"""
            background-color: {Colors.ERROR_RED};
            color: white;
            font-size: {Typography.TEXT_XS};
            font-weight: {Typography.MEDIUM};
            padding: {Spacing.XS} {Spacing.SM};
            border-radius: 12px;
            display: inline-block;
            margin: {Spacing.XS};
        """


def apply_global_styles():
    """Apply global CSS styles to the Streamlit app."""
    st.markdown(
        f"""
        <style>
        /* Global typography and layout improvements */
        .main {{
            font-family: {Typography.SANS};
            color: {Colors.TEXT_PRIMARY};
            line-height: 1.6;
        }}

        /* Improved sidebar styling */
        .css-1d391kg {{
            width: 300px !important;
            background-color: {Colors.BG_SECONDARY};
            border-right: 1px solid {Colors.BORDER_LIGHT};
        }}

        .css-1cyp50f {{
            min-width: 300px !important;
            max-width: 300px !important;
        }}

        /* Main content area styling */
        .main .block-container {{
            margin-left: 320px;
            padding-top: {Spacing.LG};
            padding-right: {Spacing.LG};
            padding-bottom: {Spacing.LG};
            max-width: none;
        }}

        /* Streamlit component improvements */
        .stButton > button {{
            transition: all 0.2s ease;
            font-family: {Typography.SANS};
            font-weight: {Typography.MEDIUM};
            border-radius: 6px;
            border: 1px solid {Colors.BORDER_DEFAULT};
            background-color: {Colors.BG_PRIMARY};
            color: {Colors.TEXT_PRIMARY};
        }}

        .stButton > button:hover {{
            border-color: {Colors.PRIMARY_BLUE};
            color: {Colors.PRIMARY_BLUE};
        }}

        .stButton > button[kind="primary"] {{
            background-color: {Colors.PRIMARY_BLUE};
            color: white;
            border-color: {Colors.PRIMARY_BLUE};
        }}

        .stButton > button[kind="primary"]:hover {{
            background-color: {Colors.PRIMARY_BLUE_DARK};
            border-color: {Colors.PRIMARY_BLUE_DARK};
        }}

        /* Select box improvements */
        .stSelectbox > div > div {{
            border-radius: 6px;
            border-color: {Colors.BORDER_DEFAULT};
        }}

        /* Number input improvements */
        .stNumberInput > div > div > input {{
            border-radius: 6px;
            border-color: {Colors.BORDER_DEFAULT};
        }}

        /* Text input improvements */
        .stTextInput > div > div > input {{
            border-radius: 6px;
            border-color: {Colors.BORDER_DEFAULT};
        }}

        /* Checkbox improvements */
        .stCheckbox > label {{
            color: {Colors.TEXT_PRIMARY};
            font-weight: {Typography.MEDIUM};
        }}

        /* Tab improvements */
        .stTabs {{
            background-color: {Colors.BG_SECONDARY};
            border-radius: 8px;
            padding: {Spacing.SM};
            margin-bottom: {Spacing.MD};
        }}

        /* Success messages */
        .stAlert[data-baseweb="notification"] {{
            border-radius: 6px;
            margin-bottom: {Spacing.MD};
        }}

        /* Metric improvements */
        .metric-container {{
            background-color: {Colors.BG_SECONDARY};
            border: 1px solid {Colors.BORDER_LIGHT};
            border-radius: 8px;
            padding: {Spacing.MD};
            margin-bottom: {Spacing.SM};
        }}

        /* Expander improvements */
        .streamlit-expanderHeader {{
            font-weight: {Typography.MEDIUM};
            color: {Colors.TEXT_PRIMARY};
        }}

        /* Code block improvements */
        .stCode {{
            font-family: {Typography.MONO};
            border-radius: 6px;
            border: 1px solid {Colors.BORDER_LIGHT};
            background-color: {Colors.BG_TERTIARY};
        }}

        /* Responsive design */
        @media (max-width: 1024px) {{
            .css-1d391kg {{
                width: 280px !important;
            }}
            .css-1cyp50f {{
                min-width: 280px !important;
                max-width: 280px !important;
            }}
            .main .block-container {{
                margin-left: 300px;
                padding-left: {Spacing.MD};
                padding-right: {Spacing.MD};
            }}
        }}

        @media (max-width: 768px) {{
            .main .block-container {{
                margin-left: 0;
                padding: {Spacing.SM};
            }}
            .css-1d391kg {{
                width: 100% !important;
            }}
        }}

        /* Hide Streamlit default elements */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stDeployButton {{display: none;}}

        /* Custom spacing classes */
        .spacing-xs {{ margin: {Spacing.XS}; }}
        .spacing-sm {{ margin: {Spacing.SM}; }}
        .spacing-md {{ margin: {Spacing.MD}; }}
        .spacing-lg {{ margin: {Spacing.LG}; }}
        .spacing-xl {{ margin: {Spacing.XL}; }}

        .spacing-top-xs {{ margin-top: {Spacing.XS}; }}
        .spacing-top-sm {{ margin-top: {Spacing.SM}; }}
        .spacing-top-md {{ margin-top: {Spacing.MD}; }}
        .spacing-top-lg {{ margin-top: {Spacing.LG}; }}
        .spacing-top-xl {{ margin-top: {Spacing.XL}; }}

        .spacing-bottom-xs {{ margin-bottom: {Spacing.XS}; }}
        .spacing-bottom-sm {{ margin-bottom: {Spacing.SM}; }}
        .spacing-bottom-md {{ margin-bottom: {Spacing.MD}; }}
        .spacing-bottom-lg {{ margin-bottom: {Spacing.LG}; }}
        .spacing-bottom-xl {{ margin-bottom: {Spacing.XL}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, icon: str = "", description: str = ""):
    """Render a professional section header."""
    header_html = f"""
    <div style="{Components.section_header_style()}">
        {icon} {title}
    </div>
    """
    if description:
        header_html += f"""
        <div style="color: {Colors.TEXT_SECONDARY}; font-size: {Typography.TEXT_SM}; margin-bottom: {Spacing.MD};">
            {description}
        </div>
        """

    st.markdown(header_html, unsafe_allow_html=True)


def render_info_card(content: str, title: str = ""):
    """Render an informational card with consistent styling."""
    card_html = f"""
    <div style="{Components.card_style()}">
        {f'<h4 style="margin: 0 0 {Spacing.SM} 0; color: {Colors.TEXT_PRIMARY};">{title}</h4>' if title else ""}
        <div style="color: {Colors.TEXT_SECONDARY};">
            {content}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_status_badge(text: str, status: str = "info"):
    """Render a status badge with appropriate color."""
    styles = {
        "info": Components.info_badge_style(),
        "success": Components.success_badge_style(),
        "warning": Components.warning_badge_style(),
        "error": Components.error_badge_style(),
    }

    badge_html = f"""
    <span style="{styles.get(status, styles["info"])}">
        {text}
    </span>
    """
    st.markdown(badge_html, unsafe_allow_html=True)
