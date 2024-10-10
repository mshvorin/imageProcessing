# Combined Image Processing Code for Problems 1, 2, and 3

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, gaussian_filter, median_filter
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd

# Function Definitions

def luminosity_method(image):
    return np.dot(image[..., :3], [0.2126, 0.7152, 0.0722]).astype(np.uint8)

def averaging_method(image):
    return np.mean(image[..., :3], axis=2).astype(np.uint8)

def lightness_method(image):
    max_rgb = np.max(image[..., :3], axis=2)
    min_rgb = np.min(image[..., :3], axis=2)
    return ((max_rgb + min_rgb) / 2).astype(np.uint8)

def scotopic_method(image):
    return np.dot(image[..., :3], [0.02, 0.02, 0.96]).astype(np.uint8)

def photopic_method(image):
    return np.dot(image[..., :3], [0.67, 0.21, 0.12]).astype(np.uint8)

def rainbow_transfer_function(gray_image):
    normalized_gray = gray_image / 255.0
    rainbow_image = plt.cm.nipy_spectral(normalized_gray)
    rgb_image = (rainbow_image[..., :3] * 255).astype(np.uint8)
    return rgb_image

def add_uniform_noise(image, low=-50, high=50):
    noise = np.random.uniform(low, high, image.shape)
    noisy_image = image + noise
    noisy_image_clipped = np.clip(noisy_image, 0, 255)
    return noisy_image_clipped.astype(np.uint8)

def add_gaussian_noise(image, mean=0, var=0.01):
    noisy_image = random_noise(image, mode='gaussian', mean=mean, var=var)
    noisy_image = np.clip(noisy_image * 255, 0, 255)
    return noisy_image.astype(np.uint8)

def add_salt_pepper_noise(image, amount=0.05):
    noisy_image = random_noise(image, mode='s&p', amount=amount)
    noisy_image = np.clip(noisy_image * 255, 0, 255)
    return noisy_image.astype(np.uint8)

def compute_psnr(original, denoised):
    return psnr(original, denoised, data_range=255)

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

# ---------------------------------------------------------------------------------------
# Problem 1: Grayscale to Color Recoloring Using Transfer Functions
# ---------------------------------------------------------------------------------------

original_image = Image.open('dog.png')
image_array = np.array(original_image)

luminosity_image = luminosity_method(image_array)
averaging_image = averaging_method(image_array)
lightness_image = lightness_method(image_array)
scotopic_image = scotopic_method(image_array)
photopic_image = photopic_method(image_array)

recolored_image = rainbow_transfer_function(luminosity_image)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.ravel()

methods = [
    ('Original Image', original_image),
    ('Luminosity Method', luminosity_image),
    ('Averaging Method', averaging_image),
    ('Lightness Method', lightness_image),
    ('Scotopic Method', scotopic_image),
    ('Photopic Method', photopic_image)
]

for i, (title, img) in enumerate(methods):
    if title == 'Original Image':
        axs[i].imshow(img)
    else:
        axs[i].imshow(img, cmap='gray')
    axs[i].set_title(title)
    axs[i].axis('off')

plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))

ax2[0].imshow(luminosity_image, cmap='gray')
ax2[0].set_title('Grayscale Image (Luminosity Method)')
ax2[0].axis('off')

ax2[1].imshow(recolored_image)
ax2[1].set_title('Recolored Image (Rainbow Transfer Function)')
ax2[1].axis('off')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------------
# Problem 2: 1D and 2D Averaging and Difference Filtering (Convolutions)
# ---------------------------------------------------------------------------------------

# Part A: 1D Averaging and Difference Filters

n = np.linspace(0, 4, 500)  # Time range adjusted to match n = 0 to 4
x = np.sin(2 * np.pi * n)   # Sine wave x[n] = sin(2 * pi * n)

h_avg = np.array([1, 1, 1]) / 3
h_diff = np.array([-1, 1])

x_avg_filtered = np.convolve(x, h_avg, mode='same')
x_diff_filtered = np.convolve(x, h_diff, mode='same')

plt.figure(figsize=(14, 6))

plt.subplot(3, 1, 1)
plt.plot(n, x, label='Original Signal')
plt.title('Original Signal')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(n, x_avg_filtered, color='orange', label='Averaging Filtered Signal')
plt.title('Signal after Averaging Filter')
plt.xlabel('n')
plt.ylabel('x_avg_filtered[n]')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(n, x_diff_filtered, color='green', label='Difference Filtered Signal')
plt.title('Signal after Difference Filter')
plt.xlabel('n')
plt.ylabel('x_diff_filtered[n]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Part B: 2D Averaging and Difference Filters

image = Image.open('dog.png').convert('L')
image_array = np.array(image)

h_avg_2d = np.ones((3, 3)) / 9
h_diff_2d = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

avg_filtered_image = convolve(image_array, h_avg_2d, mode='reflect')
diff_filtered_image = convolve(image_array, h_diff_2d, mode='reflect')

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(image_array, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(avg_filtered_image, cmap='gray')
plt.title('Image after Averaging Filter')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(diff_filtered_image, cmap='gray')
plt.title('Image after Difference Filter')
plt.axis('off')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------------
# Problem 3: Noise Models and Noise Removal
# ---------------------------------------------------------------------------------------

original_image = Image.open('dog.png').convert('L')
original_array = np.array(original_image)

uniform_noisy = add_uniform_noise(original_array)
gaussian_noisy = add_gaussian_noise(original_array / 255.0)
sp_noisy = add_salt_pepper_noise(original_array / 255.0)
gaussian_sp_noisy = add_gaussian_noise(sp_noisy / 255.0)

uniform_median = median_filter(uniform_noisy, size=3)
gaussian_median = median_filter(gaussian_noisy, size=3)
sp_median = median_filter(sp_noisy, size=3)
gaussian_sp_median = median_filter(gaussian_sp_noisy, size=3)

uniform_gaussian = gaussian_filter(uniform_noisy, sigma=1)
gaussian_gaussian = gaussian_filter(gaussian_noisy, sigma=1)
sp_gaussian = gaussian_filter(sp_noisy, sigma=1)
gaussian_sp_gaussian = gaussian_filter(gaussian_sp_noisy, sigma=1)

uniform_alpha_mean = alpha_trimmed_mean_filter(uniform_noisy)
gaussian_alpha_mean = alpha_trimmed_mean_filter(gaussian_noisy)
sp_alpha_mean = alpha_trimmed_mean_filter(sp_noisy)
gaussian_sp_alpha_mean = alpha_trimmed_mean_filter(gaussian_sp_noisy)

psnr_results = {}

noise_types = ['Uniform Distribution Noise', 'Gaussian Distribution Noise', 'Salt & Pepper Noise', 'Gaussian and Salt & Pepper Noise']
noisy_images = [uniform_noisy, gaussian_noisy, sp_noisy, gaussian_sp_noisy]
median_filtered_images = [uniform_median, gaussian_median, sp_median, gaussian_sp_median]
gaussian_filtered_images = [uniform_gaussian, gaussian_gaussian, sp_gaussian, gaussian_sp_gaussian]
alpha_mean_filtered_images = [uniform_alpha_mean, gaussian_alpha_mean, sp_alpha_mean, gaussian_sp_alpha_mean]

for idx, noise_type in enumerate(noise_types):
    psnr_results[noise_type] = {}
    psnr_noisy = compute_psnr(original_array, noisy_images[idx])
    psnr_results[noise_type]['Noisy Image'] = psnr_noisy
    psnr_median = compute_psnr(original_array, median_filtered_images[idx])
    psnr_results[noise_type]['Median Filter'] = psnr_median
    psnr_gaussian = compute_psnr(original_array, gaussian_filtered_images[idx])
    psnr_results[noise_type]['Gaussian Filter'] = psnr_gaussian
    psnr_alpha_mean = compute_psnr(original_array, alpha_mean_filtered_images[idx])
    psnr_results[noise_type]['Alpha Mean'] = psnr_alpha_mean

df = pd.DataFrame.from_dict(psnr_results, orient='index')
df = df[['Noisy Image', 'Median Filter', 'Gaussian Filter', 'Alpha Mean']]

print("Table 1: PSNR Comparison for Different Noise Models and Filters")
print(df.to_string())

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
