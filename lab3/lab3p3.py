import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd

# Load the original image
original_image = Image.open('dog.png').convert('L')  # Convert to grayscale
original_array = np.array(original_image)

# Function to add uniform noise
def add_uniform_noise(image, low=-50, high=50):
    noise = np.random.uniform(low, high, image.shape)
    noisy_image = image + noise
    noisy_image_clipped = np.clip(noisy_image, 0, 255)
    return noisy_image_clipped.astype(np.uint8)

# Function to add Gaussian noise
def add_gaussian_noise(image, mean=0, var=0.01):
    noisy_image = random_noise(image, mode='gaussian', mean=mean, var=var)
    noisy_image = np.clip(noisy_image * 255, 0, 255)
    return noisy_image.astype(np.uint8)

# Function to add salt and pepper noise
def add_salt_pepper_noise(image, amount=0.05):
    noisy_image = random_noise(image, mode='s&p', amount=amount)
    noisy_image = np.clip(noisy_image * 255, 0, 255)
    return noisy_image.astype(np.uint8)

# Function to compute PSNR
def compute_psnr(original, denoised):
    return psnr(original, denoised, data_range=255)

# Function for Alpha-Trimmed Mean Filter
def alpha_trimmed_mean_filter(image, kernel_size=3, alpha=0.25):
    padded_image = np.pad(image, pad_width=kernel_size//2, mode='reflect')
    filtered_image = np.zeros_like(image)
    m, n = image.shape
    d = int(alpha * (kernel_size ** 2))
    for i in range(m):
        for j in range(n):
            window = padded_image[i:i+kernel_size, j:j+kernel_size].flatten()
            window_sorted = np.sort(window)
            trimmed_window = window_sorted[d//2:-d//2] if d > 0 else window_sorted
            filtered_image[i, j] = np.mean(trimmed_window)
    return filtered_image.astype(np.uint8)

# Apply different types of noise
uniform_noisy = add_uniform_noise(original_array)
gaussian_noisy = add_gaussian_noise(original_array / 255.0)  # Normalize for skimage
sp_noisy = add_salt_pepper_noise(original_array / 255.0)
gaussian_sp_noisy = add_gaussian_noise(sp_noisy / 255.0)

# Apply filters
# Median Filter
uniform_median = median_filter(uniform_noisy, size=3)
gaussian_median = median_filter(gaussian_noisy, size=3)
sp_median = median_filter(sp_noisy, size=3)
gaussian_sp_median = median_filter(gaussian_sp_noisy, size=3)

# Gaussian Filter
uniform_gaussian = gaussian_filter(uniform_noisy, sigma=1)
gaussian_gaussian = gaussian_filter(gaussian_noisy, sigma=1)
sp_gaussian = gaussian_filter(sp_noisy, sigma=1)
gaussian_sp_gaussian = gaussian_filter(gaussian_sp_noisy, sigma=1)

# Alpha-Trimmed Mean Filter
uniform_alpha_mean = alpha_trimmed_mean_filter(uniform_noisy)
gaussian_alpha_mean = alpha_trimmed_mean_filter(gaussian_noisy)
sp_alpha_mean = alpha_trimmed_mean_filter(sp_noisy)
gaussian_sp_alpha_mean = alpha_trimmed_mean_filter(gaussian_sp_noisy)

# Compute PSNR values
psnr_results = {}

noise_types = ['Uniform Distribution Noise', 'Gaussian Distribution Noise', 'Salt & Pepper Noise', 'Gaussian and Salt & Pepper Noise']
noisy_images = [uniform_noisy, gaussian_noisy, sp_noisy, gaussian_sp_noisy]
median_filtered_images = [uniform_median, gaussian_median, sp_median, gaussian_sp_median]
gaussian_filtered_images = [uniform_gaussian, gaussian_gaussian, sp_gaussian, gaussian_sp_gaussian]
alpha_mean_filtered_images = [uniform_alpha_mean, gaussian_alpha_mean, sp_alpha_mean, gaussian_sp_alpha_mean]

for idx, noise_type in enumerate(noise_types):
    psnr_results[noise_type] = {}
    # Noisy Image PSNR
    psnr_noisy = compute_psnr(original_array, noisy_images[idx])
    psnr_results[noise_type]['Noisy Image'] = psnr_noisy
    # Median Filter PSNR
    psnr_median = compute_psnr(original_array, median_filtered_images[idx])
    psnr_results[noise_type]['Median Filter'] = psnr_median
    # Gaussian Filter PSNR
    psnr_gaussian = compute_psnr(original_array, gaussian_filtered_images[idx])
    psnr_results[noise_type]['Gaussian Filter'] = psnr_gaussian
    # Alpha Mean Filter PSNR
    psnr_alpha_mean = compute_psnr(original_array, alpha_mean_filtered_images[idx])
    psnr_results[noise_type]['Alpha Mean'] = psnr_alpha_mean

# Display the results in a table
df = pd.DataFrame.from_dict(psnr_results, orient='index')
df = df[['Noisy Image', 'Median Filter', 'Gaussian Filter', 'Alpha Mean']]  # Ensure correct column order

print("Table 1: PSNR Comparison for Different Noise Models and Filters")
print(df.to_string())

# Optionally, display images for visual comparison
# Here we display the results for Uniform Distribution Noise as an example
plt.figure(figsize=(12, 8))

plt.subplot(2,3,1)
plt.imshow(original_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(uniform_noisy, cmap='gray')
plt.title('Uniform Noisy Image')
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(uniform_median, cmap='gray')
plt.title('Median Filtered')
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(uniform_gaussian, cmap='gray')
plt.title('Gaussian Filtered')
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(uniform_alpha_mean, cmap='gray')
plt.title('Alpha Mean Filtered')
plt.axis('off')

plt.tight_layout()
plt.show()
