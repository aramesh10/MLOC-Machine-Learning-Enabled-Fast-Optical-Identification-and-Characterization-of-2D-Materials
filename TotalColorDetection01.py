import cv2
import colour
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

image = cv2.imread('image3.png')
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert to l*a*b colorspace

# measure substrate ~dynamically~
region_percent = 0.2  # set row based on region % of image, 0.2 =  measure 20% from top
row = int(lab_image.shape[0] * region_percent)

# measure l*a*b through given region
horizontal_line = []
for col in range(lab_image.shape[1]):
    lab_values = lab_image[row, col]
    horizontal_line.append(lab_values)

substrate_lab = np.mean(horizontal_line, axis=0)   # l*a*b mean of substrate
# print("mean l*a*b along substrate row {}: {}".format(row, substrate_lab))

# compute delta E for each pixel relative to substrate
delta_e = np.linalg.norm(lab_image - substrate_lab, axis=2)
normalized_delta_e = delta_e / np.max(delta_e) #important for visualization

#%%
# Conversions
# Perceptually Uniform Sequential Colormap
colormap = matplotlib.colormaps['plasma']
normalized_delta_e_rgb = colormap(normalized_delta_e)[:, :, :3]
bgr_plasma = cv2.cvtColor((normalized_delta_e_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# hsv colormap
colormap = matplotlib.colormaps['hsv']
normalized_delta_e_rgb = colormap(normalized_delta_e)[:, :, :3]
bgr_hsv = cv2.cvtColor((normalized_delta_e_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# Diverging Colormap
colormap = matplotlib.colormaps['Spectral']
normalized_delta_e_rgb = colormap(normalized_delta_e)[:, :, :3]
bgr_spectral = cv2.cvtColor((normalized_delta_e_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# Display the delta E maps
cv2.imshow('Original', image)
cv2.imshow('DeltaE Map', (delta_e / np.max(delta_e) * 255).astype(np.uint8))
cv2.imshow('DeltaE (Perceptually Uniform Sequential Colormap)', bgr_plasma)
cv2.imshow('DeltaE (HSV)', bgr_hsv)
cv2.imshow('DeltaE (Diverging Colormap)', bgr_spectral)

cv2.waitKey(0)
cv2.destroyAllWindows()

#%%-------------------------------------------------------
# may use this later, display map, plot the colors

# display the delta E map
plt.imshow(normalized_delta_e, cmap='plasma')
plt.colorbar(label='l*a*b* delta E')
plt.title('ΔE Map')
plt.show(block=True)

# plot the colors
# converts l*a*b* to XYZ then to xyz coordinates
xyz = colour.sRGB_to_XYZ(lab_image / 255.0)
lab_colour = colour.XYZ_to_Lab(xyz)
# plot
plt.figure(figsize=(8, 6))
plt.scatter(lab_colour[:, 1], lab_colour[:, 2], c=lab_colour[:, 0], cmap='jet')
plt.xlabel('a*')
plt.ylabel('b*')
plt.title('L*a*b* Color Space')
plt.colorbar(label='L*')
plt.grid(True)
plt.show()
