import cv2
import matplotlib.pyplot as plt
import numpy as np

# Color Processing
image = cv2.imread('dog.png')

# 1. Single Channel Extraction
B, G, R = cv2.split(image)
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
plt.savefig('singleChannelExtraction.png')
plt.show()

# 2. Color Representation
zeroes = np.zeros_like(R)
red_image = cv2.merge([zeroes, zeroes, R])
green_image = cv2.merge([zeroes, G, zeroes])
blue_image = cv2.merge([B, zeroes, zeroes])
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
plt.savefig('colorRepresentation.png')
plt.show()

# 3. HSV Channel Extraction
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv_image)
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
plt.savefig('hsvChannelExtraction.png')
plt.show()

# 4. Saturation Adjustment
def adjust_saturation(scaling_factor):
    S_scaled = np.clip(S * scaling_factor, 0, 255).astype(np.uint8)
    hsv_adjusted = cv2.merge([H, S_scaled, V])
    img_adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    return img_adjusted

factors = [0.01, 0.26, 0.46]
plt.figure(figsize=(10, 3))
for i, factor in enumerate(factors):
    plt.subplot(1, 3, i+1)
    plt.title(f'Saturation x {factor}')
    adjusted_image = adjust_saturation(factor)
    plt.imshow(cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB))
plt.savefig('saturationAdjustment.png')
plt.show()

# Image Algebra Addition
I1 = cv2.imread('plane.png', cv2.IMREAD_GRAYSCALE)
I2 = cv2.imread('car.png', cv2.IMREAD_GRAYSCALE)

if I1 is not None and I2 is not None:
    if I1.shape != I2.shape:
        I2 = cv2.resize(I2, (I1.shape[1], I1.shape[0]))

    # 1. Image Addition
    Iadd = cv2.add(I1, I2)
    plt.imshow(Iadd, cmap='gray')
    plt.title('Image Addition')
    plt.savefig('imageAddition.png')
    plt.show()

    # 2. Alpha Blending
    alpha = 0.5
    Ialpha = cv2.addWeighted(I1, alpha, I2, 1 - alpha, 0)
    plt.imshow(Ialpha, cmap='gray')
    plt.title('Alpha Blending (α = 0.5)')
    plt.savefig('alphaBlending.png')
    plt.show()
