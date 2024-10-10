import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the original image
original_image = Image.open('dog.png')

# Convert image to numpy array
image_array = np.array(original_image)

# Define grayscale conversion methods
def luminosity_method(image):
    # Using ITU-R BT.709 weights
    return np.dot(image[..., :3], [0.2126, 0.7152, 0.0722]).astype(np.uint8)

def averaging_method(image):
    return np.mean(image[..., :3], axis=2).astype(np.uint8)

def lightness_method(image):
    max_rgb = np.max(image[..., :3], axis=2)
    min_rgb = np.min(image[..., :3], axis=2)
    return ((max_rgb + min_rgb) / 2).astype(np.uint8)

def scotopic_method(image):
    # Scotopic vision weights (night vision)
    return np.dot(image[..., :3], [0.02, 0.02, 0.96]).astype(np.uint8)

def photopic_method(image):
    # Photopic vision weights (daylight vision)
    return np.dot(image[..., :3], [0.67, 0.21, 0.12]).astype(np.uint8)

# Apply grayscale methods
luminosity_image = luminosity_method(image_array)
averaging_image = averaging_method(image_array)
lightness_image = lightness_method(image_array)
scotopic_image = scotopic_method(image_array)
photopic_image = photopic_method(image_array)

# Define Rainbow transfer function
def rainbow_transfer_function(gray_image):
    # Normalize grayscale image to [0, 1]
    normalized_gray = gray_image / 255.0
    # Apply the Rainbow colormap
    rainbow_image = plt.cm.nipy_spectral(normalized_gray)
    # Convert to RGB values in range [0, 255]
    rgb_image = (rainbow_image[..., :3] * 255).astype(np.uint8)
    return rgb_image

# Apply the transfer function to the luminosity grayscale image
recolored_image = rainbow_transfer_function(luminosity_image)

# Display the grayscale images
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

# Display the recolored image alongside the grayscale image
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))

ax2[0].imshow(luminosity_image, cmap='gray')
ax2[0].set_title('Grayscale Image (Luminosity Method)')
ax2[0].axis('off')

ax2[1].imshow(recolored_image)
ax2[1].set_title('Recolored Image (Rainbow Transfer Function)')
ax2[1].axis('off')

plt.tight_layout()
plt.show()
