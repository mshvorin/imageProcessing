from PIL import Image
from scipy.ndimage import convolve
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate the 1D sine wave signal
n = np.linspace(0, 4 * np.pi, 500)
x = np.sin(2 * np.pi * n)

# Step 2: Define the 1D averaging filter
h_avg = np.array([1, 1, 1]) / 3

# Step 3: Define the 1D difference filter
h_diff = np.array([-1, 1])

# Step 4: Apply the filters using convolution
x_avg_filtered = np.convolve(x, h_avg, mode='same')
x_diff_filtered = np.convolve(x, h_diff, mode='same')

# Step 5: Plot the original and filtered signals
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

# Load the image (convert to grayscale for simplicity)
image = Image.open('dog.png').convert('L')
image_array = np.array(image)

# Define the 2D averaging filter
h_avg = np.ones((3, 3)) / 9

# Define the 2D difference filter
h_diff = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

# Apply the filters using convolution
avg_filtered_image = convolve(image_array, h_avg, mode='reflect')
diff_filtered_image = convolve(image_array, h_diff, mode='reflect')

# Display the original and filtered images
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
