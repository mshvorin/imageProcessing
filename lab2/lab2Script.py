import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_images_and_save(titles, images, save_path, figsize=(15, 5)):
    n = len(images)
    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(1, n, i + 1)
        if len(images[i].shape) == 3:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=.5)
    plt.show()

# Task 1: Noise Removal
print("Task 1: Starting noise removal")
I_original = cv2.imread('gaussiantest.png', cv2.IMREAD_GRAYSCALE)
I_original = cv2.resize(I_original, (503, 293))
n = 100
sigma = 100
noisy_images = []
for i in range(n):
    noise = np.random.normal(0, sigma, I_original.shape)
    noisy_img = np.clip(I_original.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    noisy_images.append(noisy_img)
    if i == 0:
        first_noisy_image = noisy_img.copy()

sum_images = np.sum(np.array(noisy_images, dtype=np.float32), axis=0)
I_denoised = np.clip(sum_images / n, 0, 255).astype(np.uint8)

difference = cv2.absdiff(I_original, I_denoised)
display_images_and_save(['Original Image', 'One Noisy Image', 'Denoised Image', 'Difference (For my own sanity)'],
                        [I_original, first_noisy_image, I_denoised, difference], 'Task1_Deliverable.png')

print("Task 1: Noise removal complete")

# Task 2: Wafer Defect Detection
print("Task 2: Starting wafer defect detection")
I_template = cv2.imread('Wafer_Template.png', cv2.IMREAD_GRAYSCALE)
I_defective = cv2.imread('Wafer_Noise_Defect_2.png', cv2.IMREAD_GRAYSCALE)

I_defects = cv2.subtract(I_defective, I_template)
_, I_defects_thresh = cv2.threshold(I_defects, 30, 255, cv2.THRESH_BINARY)

display_images_and_save(['Template Wafer', 'Defective Wafer', 'Detected Defects'],
                        [I_template, I_defective, I_defects_thresh], 'Task2_Deliverable.png')
print("Task 2: Defect detection complete")

# Task 3: Logarithmic Image Processing
print("Task 3: Starting logarithmic image processing")
M = 255
resized_noisy_images = [cv2.resize(img, (438, 129)) for img in noisy_images]

g1 = resized_noisy_images[0].astype(np.float32)
g2 = resized_noisy_images[1].astype(np.float32)

log_addition = g1 + g2 - (g1 * g2) / M
log_addition_clipped = np.clip(log_addition, 0, 255).astype(np.uint8)

I_denoised_task1_resized = cv2.resize(I_denoised, (438, 129))

display_images_and_save(['Logarithmic Addition', 'Denoised Image from Task 1'],
                        [log_addition_clipped, I_denoised_task1_resized], 'Task3_Deliverable.png')

print("Task 3: Logarithmic processing complete")

# Task 4: Generate Mixed Image
print("Task 4: Starting output image generation")
image_paths = ['I1.png', 'I2.png', 'I3.png', 'I4.jpg']
I1, I2, I3, I4 = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

min_height = min(I1.shape[0], I2.shape[0], I3.shape[0], I4.shape[0])
min_width = min(I1.shape[1], I2.shape[1], I3.shape[1], I4.shape[1])

I1_resized, I2_resized, I3_resized, I4_resized = [cv2.resize(img, (min_width, min_height)) for img in
                                                  [I1, I2, I3, I4]]

I_sum = cv2.add(I1_resized, I2_resized).astype(np.float32)
I3_flipped = cv2.flip(I3_resized, 0).astype(np.float32)
I_mult = I_sum * I3_flipped
I_mult_norm = cv2.normalize(I_mult, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
I_output = np.clip(I_mult_norm - I4_resized.astype(np.float32), 0, 255).astype(np.uint8)

display_images_and_save(['I1', 'I2', 'I3 Flipped', 'I4', 'Output Image'],
                        [I1_resized, I2_resized, I3_flipped.astype(np.uint8), I4_resized, I_output],
                        'Task4_Deliverable.png')

print("Task 4: Output image generated")

print("Main: All tasks completed successfully!")
