import cv2
import numpy as np
import streamlit as st
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import time
import matplotlib.pyplot as plt
import pandas as pd
import io
from scipy import ndimage
import seaborn as sns
from skimage import color, exposure, filters
from skimage.util import random_noise

# Set page config
st.set_page_config(
    page_title="Advanced Image Processing",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Utility functions
def calculate_psnr(original, filtered):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((original - filtered) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_mse(original, filtered):
    """Calculate Mean Squared Error"""
    return np.mean((original - filtered) ** 2)

def calculate_ssim(original, filtered):
    """Calculate Structural Similarity Index"""
    win_size = min(7, min(original.shape[0], original.shape[1]) // 2 * 2 + 1)
    return ssim(original, filtered, win_size=win_size, channel_axis=2, data_range=255)

def calculate_ncc(original, filtered):
    """Calculate Normalized Cross-Correlation"""
    original = original.astype(np.float64)
    filtered = filtered.astype(np.float64)
    numerator = np.sum((original - np.mean(original)) * (filtered - np.mean(filtered)))
    denominator = np.sqrt(np.sum((original - np.mean(original))**2) * np.sum((filtered - np.mean(filtered))**2))
    return numerator / denominator if denominator != 0 else 0

def calculate_computational_cost(image_shape, kernel_size, filter_type):
    """Estimate computational cost based on operation complexity"""
    pixels = image_shape[0] * image_shape[1]
    if filter_type == "Mean":
        return pixels * (kernel_size ** 2)
    elif filter_type == "Median":
        return pixels * kernel_size * kernel_size * np.log2(kernel_size * kernel_size)
    elif filter_type == "Gaussian":
        return pixels * (kernel_size ** 2) * 2  # Separable filter approximation
    return 0

def plot_histograms(original, filtered):
    """Plot RGB histograms for comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    for i, color in enumerate(['r', 'g', 'b']):
        ax1.hist(original[:,:,i].ravel(), bins=256, color=color, alpha=0.5)
    ax1.set_title('Original Image Histogram')
    ax1.set_xlim([0, 256])
    
    for i, color in enumerate(['r', 'g', 'b']):
        ax2.hist(filtered[:,:,i].ravel(), bins=256, color=color, alpha=0.5)
    ax2.set_title('Filtered Image Histogram')
    ax2.set_xlim([0, 256])
    
    return fig

def plot_quality_metrics(metrics_history):
    """Plot quality metrics over kernel sizes"""
    if len(metrics_history) < 2:
        return None
    
    df = pd.DataFrame(metrics_history)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    sns.lineplot(data=df, x='Kernel Size', y='PSNR', ax=axes[0,0])
    axes[0,0].set_title('PSNR vs Kernel Size')
    
    sns.lineplot(data=df, x='Kernel Size', y='MSE', ax=axes[0,1])
    axes[0,1].set_title('MSE vs Kernel Size')
    
    sns.lineplot(data=df, x='Kernel Size', y='SSIM', ax=axes[1,0])
    axes[1,0].set_title('SSIM vs Kernel Size')
    
    sns.lineplot(data=df, x='Kernel Size', y='Computation Time', ax=axes[1,1])
    axes[1,1].set_title('Computation Time vs Kernel Size')
    
    plt.tight_layout()
    return fig

def plot_frequency_domain(original, filtered):
    """Plot frequency domain representations"""
    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    filtered_gray = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
    
    f_original = np.fft.fft2(original_gray)
    fshift_original = np.fft.fftshift(f_original)
    magnitude_original = 20*np.log(np.abs(fshift_original))
    
    f_filtered = np.fft.fft2(filtered_gray)
    fshift_filtered = np.fft.fftshift(f_filtered)
    magnitude_filtered = 20*np.log(np.abs(fshift_filtered))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.imshow(magnitude_original, cmap='gray')
    ax1.set_title('Original Image FFT')
    ax2.imshow(magnitude_filtered, cmap='gray')
    ax2.set_title('Filtered Image FFT')
    
    return fig

# Filter functions
def apply_mean_filter(image, kernel_size):
    """Apply mean filter with border reflection"""
    return cv2.blur(image, (kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT)

def apply_median_filter(image, kernel_size):
    """Apply median filter with border reflection"""
    return cv2.medianBlur(image, kernel_size)

def apply_gaussian_filter(image, kernel_size):
    """Apply Gaussian filter with automatic sigma calculation"""
    sigma = 0.3*((kernel_size-1)*0.5 - 1) + 0.8
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)

def apply_bilateral_filter(image, kernel_size, sigma_color, sigma_space):
    """Apply bilateral filter for edge-preserving smoothing"""
    return cv2.bilateralFilter(image, kernel_size, sigma_color, sigma_space)

def apply_adaptive_filter(image, kernel_size, method='mean', threshold=30):
    """Apply adaptive thresholding based filter"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if method == 'mean':
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY_INV, kernel_size, threshold)
    else:
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, kernel_size, threshold)
    return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB)

def apply_custom_filter(image, kernel):
    """Apply custom convolution kernel"""
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)

def add_noise(image, noise_type='gaussian', amount=0.05):
    """Add various types of noise to image"""
    if noise_type == 'gaussian':
        noisy = random_noise(image, mode='gaussian', var=amount)
    elif noise_type == 'salt_pepper':
        noisy = random_noise(image, mode='s&p', amount=amount)
    elif noise_type == 'speckle':
        noisy = random_noise(image, mode='speckle', var=amount)
    else:
        noisy = image
    return (noisy * 255).astype(np.uint8)

# Edge detection functions
def detect_edges(image, method='canny', low_threshold=50, high_threshold=150):
    """Apply edge detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if method == 'canny':
        edges = cv2.Canny(gray, low_threshold, high_threshold)
    elif method == 'sobel':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = (edges / edges.max() * 255).astype(np.uint8)
    elif method == 'laplacian':
        edges = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        edges = np.absolute(edges)
        edges = (edges / edges.max() * 255).astype(np.uint8)
    else:
        edges = gray
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

# Streamlit App
st.title("🖼️ Advanced Image Enhancement and Filtering Analysis")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "tiff", "bmp"])
    
    if uploaded_file is None:
        demo_option = st.selectbox("Or use demo image", 
                                 ["None", "Landscape", "Portrait", "Medical", "Text", "Low-light"])
    else:
        demo_option = "None"
    
    st.subheader("Noise Parameters")
    add_noise_option = st.checkbox("Add artificial noise")
    noise_type = st.selectbox("Noise type", ["gaussian", "salt_pepper", "speckle"], disabled=not add_noise_option)
    noise_amount = st.slider("Noise amount", 0.01, 0.5, 0.05, 0.01, disabled=not add_noise_option)
    
    st.subheader("Filter Parameters")
    filter_type = st.selectbox("Filter Type", 
                             ["Mean", "Median", "Gaussian", "Bilateral", "Adaptive", "Custom"])
    
    kernel_size = st.slider("Kernel Size", 3, 31, 3, step=2)
    
    if filter_type == "Bilateral":
        sigma_color = st.slider("Sigma Color", 1, 100, 25)
        sigma_space = st.slider("Sigma Space", 1, 100, 25)
    elif filter_type == "Adaptive":
        threshold = st.slider("Threshold", 1, 100, 30)
        adaptive_method = st.selectbox("Adaptive Method", ["mean", "gaussian"])
    elif filter_type == "Custom":
        custom_kernel = np.zeros((3,3), dtype=np.float32)
        cols = st.columns(3)
        for i in range(3):
            for j in range(3):
                custom_kernel[i,j] = cols[j].number_input(f"Kernel[{i},{j}]", -10.0, 10.0, 0.0, 0.1)
        custom_kernel /= custom_kernel.sum() if custom_kernel.sum() != 0 else 1
    
    st.subheader("Edge Detection")
    edge_detection = st.checkbox("Apply Edge Detection")
    if edge_detection:
        edge_method = st.selectbox("Edge Method", ["canny", "sobel", "laplacian"])
        if edge_method == "canny":
            low_thresh = st.slider("Low Threshold", 1, 255, 50)
            high_thresh = st.slider("High Threshold", 1, 255, 150)
    
    st.subheader("Analysis Options")
    show_histograms = st.checkbox("Show Histograms", True)
    show_frequency = st.checkbox("Show Frequency Domain", False)
    show_metrics = st.checkbox("Show Detailed Metrics", True)
    compare_kernels = st.checkbox("Compare Kernel Sizes", False)

# Load image
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
elif demo_option != "None":
    if demo_option == "Landscape":
        image = np.array(Image.open("landscape_demo.jpg"))  # Replace with actual path
    elif demo_option == "Portrait":
        image = np.array(Image.open("portrait_demo.jpg"))
    elif demo_option == "Medical":
        image = np.array(Image.open("medical_demo.jpg"))
    elif demo_option == "Text":
        image = np.array(Image.open("text_demo.jpg"))
    elif demo_option == "Low-light":
        image = np.array(Image.open("lowlight_demo.jpg"))
else:
    st.info("Please upload an image or select a demo image")
    st.stop()

# Convert to RGB if needed
if len(image.shape) == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
elif image.shape[2] == 4:
    image = image[:,:,:3]

# Add noise if requested
if add_noise_option:
    original_image = image.copy()
    image = add_noise(image, noise_type, noise_amount)
else:
    original_image = image.copy()

# Main processing
col1, col2 = st.columns(2)

with col1:
    st.image(original_image, caption='Original Image', use_column_width=True)

with col2:
    start_time = time.perf_counter()
    
    if filter_type == "Mean":
        filtered_image = apply_mean_filter(image, kernel_size)
    elif filter_type == "Median":
        filtered_image = apply_median_filter(image, kernel_size)
    elif filter_type == "Gaussian":
        filtered_image = apply_gaussian_filter(image, kernel_size)
    elif filter_type == "Bilateral":
        filtered_image = apply_bilateral_filter(image, kernel_size, sigma_color, sigma_space)
    elif filter_type == "Adaptive":
        filtered_image = apply_adaptive_filter(image, kernel_size, adaptive_method, threshold)
    elif filter_type == "Custom":
        filtered_image = apply_custom_filter(image, custom_kernel)
    
    if edge_detection:
        if edge_method == "canny":
            edges = detect_edges(filtered_image, edge_method, low_thresh, high_thresh)
        else:
            edges = detect_edges(filtered_image, edge_method)
        filtered_image = cv2.addWeighted(filtered_image, 0.8, edges, 0.2, 0)
    
    end_time = time.perf_counter()
    computation_time = end_time - start_time
    
    st.image(filtered_image, caption='Processed Image', use_column_width=True)

# Metrics calculation
metrics = {
    "PSNR": calculate_psnr(original_image, filtered_image),
    "MSE": calculate_mse(original_image, filtered_image),
    "SSIM": calculate_ssim(original_image, filtered_image),
    "NCC": calculate_ncc(original_image, filtered_image),
    "Computation Time": computation_time,
    "Computational Cost": calculate_computational_cost(image.shape, kernel_size, filter_type)
}

# Display metrics
if show_metrics:
    st.subheader("Quality Metrics")
    metric_cols = st.columns(4)
    metric_cols[0].metric("PSNR", f"{metrics['PSNR']:.2f}")
    metric_cols[1].metric("MSE", f"{metrics['MSE']:.2f}")
    metric_cols[2].metric("SSIM", f"{metrics['SSIM']:.3f}")
    metric_cols[3].metric("NCC", f"{metrics['NCC']:.3f}")
    
    st.write(f"**Computation Time:** {metrics['Computation Time']:.4f} seconds")
    st.write(f"**Estimated Computational Cost:** {metrics['Computational Cost']:,} operations")

# Visualization tabs
tab1, tab2, tab3, tab4 = st.tabs(["Histograms", "Frequency", "Kernel Analysis", "Documentation"])

with tab1:
    if show_histograms:
        st.pyplot(plot_histograms(original_image, filtered_image))

with tab2:
    if show_frequency:
        st.pyplot(plot_frequency_domain(original_image, filtered_image))

with tab3:
    if compare_kernels:
        st.subheader("Kernel Size Comparison")
        
        kernel_sizes = list(range(3, 32, 2))
        metrics_history = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ks in enumerate(kernel_sizes):
            progress_bar.progress((i + 1) / len(kernel_sizes))
            status_text.text(f"Processing kernel size {ks}...")
            
            start_time = time.perf_counter()
            
            if filter_type == "Mean":
                current_filtered = apply_mean_filter(image, ks)
            elif filter_type == "Median":
                current_filtered = apply_median_filter(image, ks)
            elif filter_type == "Gaussian":
                current_filtered = apply_gaussian_filter(image, ks)
            
            computation_time = time.perf_counter() - start_time
            
            metrics_history.append({
                "Kernel Size": ks,
                "PSNR": calculate_psnr(original_image, current_filtered),
                "MSE": calculate_mse(original_image, current_filtered),
                "SSIM": calculate_ssim(original_image, current_filtered),
                "Computation Time": computation_time
            })
        
        progress_bar.empty()
        status_text.empty()
        
        st.pyplot(plot_quality_metrics(metrics_history))
        
        df = pd.DataFrame(metrics_history)
        st.dataframe(df.style.highlight_max(subset=['PSNR', 'SSIM'], color='lightgreen')
                    .highlight_min(subset=['MSE', 'Computation Time'], color='lightgreen'))

with tab4:
    st.subheader("Filter Documentation")
    
    if filter_type == "Mean":
        st.markdown("""
        **Mean Filter (Box Filter)**
        - Simple averaging of pixel values in the kernel window
        - Effective for Gaussian noise but blurs edges
        - Computational complexity: O(n*k²) where n is number of pixels, k is kernel size
        """)
    elif filter_type == "Median":
        st.markdown("""
        **Median Filter**
        - Replaces pixel value with median of neighboring pixels
        - Excellent for salt-and-pepper noise while preserving edges
        - Computational complexity: O(n*k² log k) due to sorting
        """)
    elif filter_type == "Gaussian":
        st.markdown("""
        **Gaussian Filter**
        - Weighted average with Gaussian distribution weights
        - Better for Gaussian noise with smoother transitions
        - Can be implemented separably for O(n*k) complexity
        """)
    elif filter_type == "Bilateral":
        st.markdown("""
        **Bilateral Filter**
        - Edge-preserving smoothing filter
        - Combines spatial and intensity domain weighting
        - Computational complexity: O(n*k²)
        """)
    
    st.subheader("Performance Tips")
    st.markdown("""
    - Start with small kernel sizes (3x3 or 5x5)
    - For large images, consider downsampling first
    - Median filter is computationally expensive for large kernels
    - Gaussian filter can be separated into 1D operations for speed
    """)

# Use-case scenarios
st.header("Practical Applications")
expander = st.expander("Show Application Scenarios")
with expander:
    st.markdown("""
    **1. Medical Imaging (Recommended: Median Filter)**
    - Remove salt-and-pepper noise from MRI/CT scans
    - Preserve important edges and structures
    - Typical kernel size: 3x3 to 7x7
    
    **2. Photography Enhancement (Recommended: Bilateral Filter)**
    - Smooth skin tones while preserving facial features
    - Reduce noise in low-light conditions
    - Typical parameters: σ_color=25-75, σ_space=5-25
    
    **3. Document Processing (Recommended: Adaptive Threshold)**
    - Binarize scanned documents with uneven lighting
    - Enhance text readability
    - Typical kernel size: 15-31, threshold=10-30
    
    **4. Satellite Imagery (Recommended: Gaussian Filter)**
    - Smooth atmospheric noise
    - Prepare images for feature detection
    - Typical kernel size: 5x5 to 9x9
    
    **5. Real-time Video Processing (Recommended: Mean Filter)**
    - Fast noise reduction for surveillance systems
    - Balance between quality and performance
    - Typical kernel size: 3x3 to 5x5
    """)

# Export results
st.sidebar.header("Export Results")
if st.sidebar.button("Save Processed Image"):
    processed_img = Image.fromarray(filtered_image)
    buf = io.BytesIO()
    processed_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.sidebar.download_button(
        label="Download Image",
        data=byte_im,
        file_name="processed_image.png",
        mime="image/png"
    )

if st.sidebar.button("Save Metrics Report"):
    report = f"""
    Image Processing Report
    ======================
    
    Original Image Dimensions: {original_image.shape[1]}x{original_image.shape[0]}
    Filter Type: {filter_type}
    Kernel Size: {kernel_size}x{kernel_size}
    
    Quality Metrics:
    - PSNR: {metrics['PSNR']:.2f} dB
    - MSE: {metrics['MSE']:.2f}
    - SSIM: {metrics['SSIM']:.3f}
    - NCC: {metrics['NCC']:.3f}
    
    Performance:
    - Computation Time: {metrics['Computation Time']:.4f} seconds
    - Estimated Operations: {metrics['Computational Cost']:,}
    """
    st.sidebar.download_button(
        label="Download Report",
        data=report,
        file_name="image_processing_report.txt",
        mime="text/plain"
    )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Advanced Image Processing Tool**  
Version 2.1  
""")