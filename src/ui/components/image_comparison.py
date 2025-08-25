"""Advanced image comparison tools and utilities."""

import numpy as np
import streamlit as st
from PIL import Image, ImageChops, ImageStat
from skimage.metrics import structural_similarity as ssim

from ui.utils.image_utils import optimize_image_for_display


def calculate_image_comparison(image_a: Image.Image, image_b: Image.Image) -> dict:
    """
    Calculate comparison metrics between two images.

    Args:
        image_a: First image for comparison
        image_b: Second image for comparison

    Returns:
        Dictionary with comparison metrics
    """
    # Ensure images are the same size
    if image_a.size != image_b.size:
        # Resize to smaller dimensions
        min_width = min(image_a.width, image_b.width)
        min_height = min(image_a.height, image_b.height)
        image_a = image_a.resize((min_width, min_height), Image.Resampling.LANCZOS)
        image_b = image_b.resize((min_width, min_height), Image.Resampling.LANCZOS)

    # Convert to same mode if needed
    if image_a.mode != image_b.mode:
        if image_a.mode == "RGBA" or image_b.mode == "RGBA":
            image_a = image_a.convert("RGBA")
            image_b = image_b.convert("RGBA")
        else:
            image_a = image_a.convert("RGB")
            image_b = image_b.convert("RGB")

    # Convert to numpy arrays
    array_a = np.array(image_a)
    array_b = np.array(image_b)

    # Calculate MSE
    mse = np.mean((array_a.astype(float) - array_b.astype(float)) ** 2)

    # Calculate PSNR
    if mse == 0:
        psnr = float("inf")
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    # Calculate SSIM for grayscale comparison
    if len(array_a.shape) == 3:  # Color image
        # Convert to grayscale for SSIM
        gray_a = np.dot(array_a[..., :3], [0.2989, 0.5870, 0.1140])
        gray_b = np.dot(array_b[..., :3], [0.2989, 0.5870, 0.1140])
    else:  # Grayscale
        gray_a = array_a
        gray_b = array_b

    try:
        ssim_score = ssim(gray_a, gray_b, data_range=gray_a.max() - gray_a.min())
    except Exception:
        ssim_score = 0.0

    # Create difference image
    diff_image = ImageChops.difference(image_a.convert("RGB"), image_b.convert("RGB"))

    return {"mse": mse, "psnr": psnr, "ssim_score": ssim_score, "diff_image": diff_image}


def render_comparison_analysis():
    """Render detailed comparison analysis between images."""

    if not (st.session_state.get("original_image") and st.session_state.get("processed_image")):
        st.info("Need both original and processed images for comparison analysis")
        return

    st.subheader("🔍 Advanced Comparison Analysis")

    original = st.session_state.original_image
    processed = st.session_state.processed_image

    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Statistics", "🎨 Histogram", "📏 Difference", "🎯 Quality"]
    )

    with tab1:
        render_statistical_comparison(original, processed)

    with tab2:
        render_histogram_comparison(original, processed)

    with tab3:
        render_difference_analysis(original, processed)

    with tab4:
        render_quality_metrics(original, processed)


def render_statistical_comparison(original: Image.Image, processed: Image.Image):
    """Render statistical comparison between images."""

    st.write("### 📊 Statistical Analysis")

    # Ensure images are the same size for comparison
    if original.size != processed.size:
        st.warning("Images have different dimensions - statistics may not be directly comparable")

        # Resize processed to match original for analysis
        processed_resized = processed.resize(original.size, Image.Resampling.LANCZOS)
    else:
        processed_resized = processed

    # Convert to same mode if needed
    if original.mode != processed_resized.mode:
        if original.mode == "RGB" and processed_resized.mode == "RGBA":
            original_compare = original.convert("RGBA")
            processed_compare = processed_resized
        elif original.mode == "RGBA" and processed_resized.mode == "RGB":
            original_compare = original
            processed_compare = processed_resized.convert("RGBA")
        else:
            # Default conversion
            original_compare = original.convert("RGB")
            processed_compare = processed_resized.convert("RGB")
    else:
        original_compare = original
        processed_compare = processed_resized

    # Get statistics
    orig_stats = ImageStat.Stat(original_compare)
    proc_stats = ImageStat.Stat(processed_compare)

    # Display channel statistics
    channels = (
        ["Red", "Green", "Blue"]
        if original_compare.mode == "RGB"
        else ["Red", "Green", "Blue", "Alpha"]
    )

    for i, channel in enumerate(channels[: len(orig_stats.mean)]):
        st.write(f"**{channel} Channel:**")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Original Mean", f"{orig_stats.mean[i]:.1f}")
            st.metric(
                "Processed Mean",
                f"{proc_stats.mean[i]:.1f}",
                delta=f"{proc_stats.mean[i] - orig_stats.mean[i]:+.1f}",
            )

        with col2:
            st.metric("Original StdDev", f"{orig_stats.stddev[i]:.1f}")
            st.metric(
                "Processed StdDev",
                f"{proc_stats.stddev[i]:.1f}",
                delta=f"{proc_stats.stddev[i] - orig_stats.stddev[i]:+.1f}",
            )

        with col3:
            st.metric("Original Min", f"{orig_stats.extrema[i][0]}")
            st.metric(
                "Processed Min",
                f"{proc_stats.extrema[i][0]}",
                delta=f"{proc_stats.extrema[i][0] - orig_stats.extrema[i][0]:+}",
            )

        with col4:
            st.metric("Original Max", f"{orig_stats.extrema[i][1]}")
            st.metric(
                "Processed Max",
                f"{proc_stats.extrema[i][1]}",
                delta=f"{proc_stats.extrema[i][1] - orig_stats.extrema[i][1]:+}",
            )


def render_histogram_comparison(original: Image.Image, processed: Image.Image):
    """Render histogram comparison (placeholder for now)."""

    st.write("### 🎨 Histogram Comparison")

    # This would require matplotlib or plotly for proper implementation
    st.info("📊 Histogram comparison would show RGB channel distributions")
    st.info("🔍 This feature requires additional visualization libraries")

    # Basic histogram data
    try:
        # Get basic histogram info
        orig_hist = original.histogram()
        proc_hist = processed.histogram()

        st.write(f"**Original histogram bins:** {len(orig_hist)}")
        st.write(f"**Processed histogram bins:** {len(proc_hist)}")

        if len(orig_hist) == len(proc_hist):
            # Calculate histogram difference
            hist_diff = sum(abs(a - b) for a, b in zip(orig_hist, proc_hist, strict=False))
            total_pixels = original.width * original.height
            normalized_diff = hist_diff / (total_pixels * 255) if total_pixels > 0 else 0

            st.metric(
                "Histogram Difference",
                f"{normalized_diff:.3f}",
                help="Normalized difference between histograms (0 = identical, 1 = completely different)",
            )

    except Exception as e:
        st.error(f"Error calculating histogram: {str(e)}")


def render_difference_analysis(original: Image.Image, processed: Image.Image):
    """Render visual difference analysis."""

    st.write("### 📏 Difference Analysis")

    # Ensure images are comparable
    if original.size != processed.size:
        st.warning("Resizing processed image to match original for difference calculation")
        processed_resized = processed.resize(original.size, Image.Resampling.LANCZOS)
    else:
        processed_resized = processed

    # Convert to same mode
    if original.mode != processed_resized.mode:
        common_mode = "RGB" if "A" not in original.mode else "RGBA"
        original_compare = original.convert(common_mode)
        processed_compare = processed_resized.convert(common_mode)
    else:
        original_compare = original
        processed_compare = processed_resized

    try:
        # Calculate difference image
        diff_image = ImageChops.difference(original_compare, processed_compare)

        # Display difference image
        st.write("**Visual Difference Map:**")
        diff_display = optimize_image_for_display(diff_image, 600, 400)
        st.image(
            diff_display,
            caption="Difference Image (white = no change, colored = changed)",
            use_container_width=True,
        )

        # Calculate difference statistics
        diff_stats = ImageStat.Stat(diff_image)

        col1, col2, col3 = st.columns(3)

        with col1:
            avg_diff = sum(diff_stats.mean) / len(diff_stats.mean)
            st.metric(
                "Average Difference",
                f"{avg_diff:.1f}/255",
                help="Average pixel difference across all channels",
            )

        with col2:
            max_diff = max(max(extrema) for extrema in diff_stats.extrema)
            st.metric(
                "Maximum Difference", f"{max_diff}/255", help="Largest pixel difference found"
            )

        with col3:
            # Calculate percentage of changed pixels
            diff_array = np.array(diff_image)
            if len(diff_array.shape) == 3:
                changed_pixels = np.any(diff_array > 0, axis=2)
            else:
                changed_pixels = diff_array > 0

            changed_percent = np.sum(changed_pixels) / changed_pixels.size * 100
            st.metric(
                "Pixels Changed",
                f"{changed_percent:.1f}%",
                help="Percentage of pixels that changed",
            )

    except Exception as e:
        st.error(f"Error calculating difference: {str(e)}")


def render_quality_metrics(original: Image.Image, processed: Image.Image):
    """Render quality comparison metrics."""

    st.write("### 🎯 Quality Metrics")

    # Basic quality indicators
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Original Image Quality:**")
        render_single_image_quality(original)

    with col2:
        st.write("**Processed Image Quality:**")
        render_single_image_quality(processed)

    # Comparative quality analysis
    st.write("**Comparative Analysis:**")

    # Resolution comparison
    orig_megapixels = (original.width * original.height) / 1_000_000
    proc_megapixels = (processed.width * processed.height) / 1_000_000

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Resolution",
            f"{proc_megapixels:.2f} MP",
            delta=f"{proc_megapixels - orig_megapixels:+.2f} MP",
        )

    with col2:
        # Aspect ratio comparison
        orig_ratio = original.width / original.height
        proc_ratio = processed.width / processed.height
        ratio_change = abs(proc_ratio - orig_ratio)

        st.metric(
            "Aspect Ratio",
            f"{proc_ratio:.2f}:1",
            delta=f"±{ratio_change:.3f}" if ratio_change > 0.001 else "Preserved",
        )

    with col3:
        # Color depth comparison
        orig_colors = len(original.getcolors(maxcolors=256 * 256 * 256) or [])
        proc_colors = len(processed.getcolors(maxcolors=256 * 256 * 256) or [])

        st.metric("Unique Colors", f"{proc_colors:,}", delta=f"{proc_colors - orig_colors:+,}")


def render_single_image_quality(image: Image.Image):
    """Render quality metrics for a single image."""

    # Basic metrics
    megapixels = (image.width * image.height) / 1_000_000
    aspect_ratio = image.width / image.height

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Resolution", f"{megapixels:.2f} MP")
        st.metric("Aspect Ratio", f"{aspect_ratio:.2f}:1")

    with col2:
        st.metric("Color Mode", image.mode)

        # Estimate dynamic range
        try:
            stats = ImageStat.Stat(image)
            if len(stats.extrema) > 0:
                # Calculate average dynamic range across channels
                dynamic_ranges = [(max_val - min_val) for min_val, max_val in stats.extrema]
                avg_dynamic_range = sum(dynamic_ranges) / len(dynamic_ranges)
                st.metric("Dynamic Range", f"{avg_dynamic_range:.0f}/255")
        except Exception:
            st.metric("Dynamic Range", "N/A")


def calculate_image_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """Calculate similarity score between two images."""

    # Ensure images are comparable
    if img1.size != img2.size:
        img2_resized = img2.resize(img1.size, Image.Resampling.LANCZOS)
    else:
        img2_resized = img2

    # Convert to same mode
    if img1.mode != img2_resized.mode:
        common_mode = "RGB"
        img1_compare = img1.convert(common_mode)
        img2_compare = img2_resized.convert(common_mode)
    else:
        img1_compare = img1
        img2_compare = img2_resized

    try:
        # Calculate difference
        diff = ImageChops.difference(img1_compare, img2_compare)
        diff_stats = ImageStat.Stat(diff)

        # Calculate similarity (1.0 = identical, 0.0 = completely different)
        avg_diff = sum(diff_stats.mean) / len(diff_stats.mean)
        similarity = 1.0 - (avg_diff / 255.0)

        return max(0.0, min(1.0, similarity))

    except Exception:
        return 0.0


def render_similarity_score():
    """Render similarity score between original and processed images."""

    if not (st.session_state.get("original_image") and st.session_state.get("processed_image")):
        return

    similarity = calculate_image_similarity(
        st.session_state.original_image, st.session_state.processed_image
    )

    st.metric(
        "Image Similarity",
        f"{similarity:.1%}",
        help="Similarity between original and processed images (100% = identical)",
    )

    # Visual similarity indicator
    if similarity > 0.95:
        st.success("🟢 Very similar - minimal changes")
    elif similarity > 0.8:
        st.info("🟡 Moderately similar - noticeable changes")
    elif similarity > 0.5:
        st.warning("🟠 Quite different - significant changes")
    else:
        st.error("🔴 Very different - major transformation")
