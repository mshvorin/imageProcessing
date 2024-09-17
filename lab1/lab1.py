import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

file_path = '/home/daedalus/digitalImage/lab1/dog.png'

# Load the color image
image = cv2.imread(file_path)

# Split the image into RGB channels
B, G, R = cv2.split(image)

# Display each channel as grayscale
plt.figure(figsize=(10, 3))

plt.subplot(1, 3, 1)
plt.title('Red Channel')
plt.imshow(R, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Green Channel')
plt.imshow(G, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Blue Channel')
plt.imshow(B, cmap='gray')

plt.show()

zeroes = np.zeros_like(R)

# Merge channels for red, green, blue visualizations
red_image = cv2.merge([zeroes, zeroes, R])
green_image = cv2.merge([zeroes, G, zeroes])
blue_image = cv2.merge([B, zeroes, zeroes])

# Display the false color images
plt.figure(figsize=(10, 3))

plt.subplot(1, 3, 1)
plt.title('Red')
plt.imshow(cv2.cvtColor(red_image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title('Green')
plt.imshow(cv2.cvtColor(green_image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 3)
plt.title('Blue')
plt.imshow(cv2.cvtColor(blue_image, cv2.COLOR_BGR2RGB))

plt.show()

# Convert the image to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv_image)

# Display the HSV channels
plt.figure(figsize=(10, 3))

plt.subplot(1, 3, 1)
plt.title('Hue Channel')
plt.imshow(H, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Saturation Channel')
plt.imshow(S, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Value Channel')
plt.imshow(V, cmap='gray')

plt.show()

def adjust_saturation(scaling_factor):
    # Adjust the saturation channel
    S_scaled = np.clip(S * scaling_factor, 0, 255).astype(np.uint8)
    
    # Recombine with H and V channels
    hsv_adjusted = cv2.merge([H, S_scaled, V])
    img_adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    
    return img_adjusted

# Display images with different saturation adjustments
factors = [0.01, 0.26, 0.46]
plt.figure(figsize=(10, 3))

for i, factor in enumerate(factors):
    plt.subplot(1, 3, i+1)
    plt.title(f'Saturation x {factor}')
    adjusted_image = adjust_saturation(factor)
    plt.imshow(cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB))

plt.show()

# Load the grayscale images
I1 = cv2.imread('/home/daedalus/digitalImage/lab1/plane.png', cv2.IMREAD_GRAYSCALE)
I2 = cv2.imread('/home/daedalus/digitalImage/lab1/car.png', cv2.IMREAD_GRAYSCALE)

if I1 is None or I2 is None:
    print("Error: One or both images could not be loaded.")
else:
    # Resize I2 to match I1's dimensions if they're not the same
    if I1.shape != I2.shape:
        I2 = cv2.resize(I2, (I1.shape[1], I1.shape[0]))

    # Perform image addition and clip values to remain within [0, 255]
    Iadd = cv2.add(I1, I2)

    # Display the result of image addition
    plt.imshow(Iadd, cmap='gray')
    plt.title('Image Addition')
    plt.show()

    # Perform alpha blending with alpha = 0.5
    alpha = 0.5
    Ialpha = cv2.addWeighted(I1, alpha, I2, 1 - alpha, 0)

    # Display the blended image
    plt.imshow(Ialpha, cmap='gray')
    plt.title('Alpha Blending (α = 0.5)')
    plt.show()
