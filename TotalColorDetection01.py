import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image2.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the grayscale image
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_image = clahe.apply(blurred_image)

# Compute histogram for the enhanced image
hist = cv2.calcHist([enhanced_image], [0], None, [256], [0, 256])

# Find peaks in the histogram
peaks = np.where(hist > np.max(hist) * 0.1)[0]  # Adjust the threshold (0.1) as needed

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Calculate threshold values
low_intensity = np.min(peaks)
high_intensity = np.max(peaks)

# Define medium intensity levels (you may adjust these based on your specific requirements)
medium_intensity_1 = low_intensity + (high_intensity - low_intensity) // 4
medium_intensity_2 = low_intensity + (high_intensity - low_intensity) // 2
medium_intensity_3 = low_intensity + 3 * (high_intensity - low_intensity) // 4

# Define a color mapping function based on intensity
def map_intensity_to_color(intensity, low_intensity, medium_intensity_1, medium_intensity_2, medium_intensity_3, high_intensity):
    # Map intensity levels to colors based on low, medium, and high intensity ranges
    if intensity <= low_intensity:
        return (153, 153, 255)  # Blue for low intensity
    elif intensity < medium_intensity_1:
        return (150, 255, 150)  # Green for medium intensity 1
    elif intensity < medium_intensity_2:
        return (255, 150, 150)  # Red for medium intensity 2
    elif intensity < medium_intensity_3:
        return (150, 255, 255)  # Yellow for medium intensity 3
    else:
        return (255, 255, 255)  # White for high intensity

# Apply color mapping to each pixel of the enhanced grayscale image
colorized_image = np.zeros((enhanced_image.shape[0], enhanced_image.shape[1], 3), dtype=np.uint8)
for y in range(enhanced_image.shape[0]):
    for x in range(enhanced_image.shape[1]):
        intensity = enhanced_image[y, x]
        color = map_intensity_to_color(intensity, low_intensity, medium_intensity_1, medium_intensity_2, medium_intensity_3, high_intensity)
        colorized_image[y, x] = color



#PLOT
hue_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
saturation_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
value_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

# Plot histograms
plt.figure(figsize=(15, 5))

# Plot Hue histogram
plt.subplot(1, 3, 1)
plt.plot(hue_hist, color='b')
plt.xlim([0, 180])
plt.title('Histogram of Hue channel')
plt.xlabel('Hue value')
plt.ylabel('Frequency')

# Plot Saturation histogram
plt.subplot(1, 3, 2)
plt.plot(saturation_hist, color='g')
plt.xlim([0, 256])
plt.title('Histogram of Saturation channel')
plt.xlabel('Saturation value')
plt.ylabel('Frequency')

# Plot Value histogram
plt.subplot(1, 3, 3)
plt.plot(value_hist, color='r')
plt.xlim([0, 256])
plt.title('Histogram of Value channel')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

#PLOT


# Display the colorized image
cv2.imshow('Colorized Image with Enhanced Contrast', colorized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


