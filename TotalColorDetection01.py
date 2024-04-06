import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('7_data0005_normalized.jpg')
enhanced_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#find histogram
hist = cv2.calcHist([enhanced_image], [0], None, [256], [0, 256])
#histogram peaks
peaks = np.where(hist > np.max(hist) * 0.1)[0]  # Adjust the threshold (0.1) as needed

#define intensity levels based on histogram peaks
low_intensity = np.min(peaks)
high_intensity = np.max(peaks)
medium_intensity_1 = low_intensity + (high_intensity - low_intensity) // 4
medium_intensity_2 = low_intensity + (high_intensity - low_intensity) // 2
medium_intensity_3 = low_intensity + 3 * (high_intensity - low_intensity) // 4

#color mapping function based on intensity
def map_intensity_to_color(intensity, low_intensity, medium_intensity_1, medium_intensity_2, medium_intensity_3, high_intensity):
    if intensity >= low_intensity and intensity < medium_intensity_1:
        return (0, 204, 255)  # Blue for low intensity
    elif intensity >= medium_intensity_1 and intensity < medium_intensity_2:
        return (51, 153, 102)  # Green for medium intensity 1
    elif intensity >= medium_intensity_2 and intensity < medium_intensity_3:
        return (128, 0, 0)  # Red for medium intensity 2
    else:
        return (128, 0, 128)  # purple for high intensity

# Apply color mapping to each pixel
colorized_image = np.zeros((enhanced_image.shape[0], enhanced_image.shape[1], 3), dtype=np.uint8)
for y in range(enhanced_image.shape[0]):
    for x in range(enhanced_image.shape[1]):
        intensity = enhanced_image[y, x]
        color = map_intensity_to_color(intensity, low_intensity, medium_intensity_1, medium_intensity_2, medium_intensity_3, high_intensity)
        colorized_image[y, x] = color

cv2.imshow('Colorized Image', colorized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#PLOT
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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


