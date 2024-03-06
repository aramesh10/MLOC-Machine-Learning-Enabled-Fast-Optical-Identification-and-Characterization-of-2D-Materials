import cv2
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread('graphene_72nm_Al2O3.png', cv2.IMREAD_GRAYSCALE)

#pixel ranges
column = 192
start_row = 10
end_row = 85

# initialize arrays to store values
contrast = []
row_positions = []

# measure contrast for each row in the specified range and column
for row_index in range(start_row, end_row + 1):
    # Select the specific pixel along the row in the column
    pixel_select = image[row_index, column]

    # Store the contrast value and row position
    contrast.append(pixel_select)
    row_positions.append(row_index)

    print(f"Contrast at Row {row_index}: {pixel_select}")
    print (f"Contrast Ratio at {row_index}: {c_ratio}")
# Plot the measured contrast throughout the row range
plt.figure(figsize=(10, 5))
plt.plot(row_positions, contrast, marker='o', linestyle='-', color='b')
plt.xlabel('Row Position')
plt.ylabel('Contrast Value')
plt.title(f'Change in Contrast Across Pixel Row, Column: {column}')
plt.grid(True)
plt.show()
