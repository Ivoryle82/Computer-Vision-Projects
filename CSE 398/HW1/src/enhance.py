import os
import cv2
import numpy as np

def convolve(image, kernel):
    (iH, iW, iD) = image.shape[:3]
    (kH, kW) = kernel.shape[:2]

    padW = (kW - 1) // 2
    padH = (kH - 1) // 2
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW, iD), dtype="float32")

    for y in np.arange(padH, iH + padH):
        for x in np.arange(padW, iW + padW):
            for z in range(3):
                roi = image[y - padH:y + padH + 1, x - padW:x + padW + 1, z]
                k = (roi * kernel).sum()
                output[y - padH, x - padW, z] = k

    return output

def rescale_intensity(img, in_range=(0, 255)):
    min_val, max_val = in_range
    img = np.clip(img, min_val, max_val)
    return img

def display_save(title, img, output_path):
    # Rescale intensity
    img_rescaled = rescale_intensity(img, in_range=(0, 255))
    
    # Convert to uint8
    img_uint8 = img_rescaled.astype("uint8")
    
    # Display image
    cv2.imshow(title, img_uint8)
    cv2.waitKey(0)
    
    output_folder = os.path.dirname(output_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Save image
    cv2.imwrite(output_path, img_uint8)
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Gaussian filter kernel
gaussian_filter = np.array((
    [1,  2,  1,  2,  1],
    [2,  4,  8,  4,  2],
    [1,  8, 16,  8,  1],
    [2,  4,  8,  4,  2],
    [1,  2,  1,  2,  1]), dtype="float") / 64  # Adjusted kernel values and normalization

# Specify the path to the folder containing the images
input_folder = '/Users/ivoryle/Desktop/CSE 398/HW1/data/input1'
output_folder = '/Users/ivoryle/Desktop/CSE 398/HW1/output'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Specify the subfolder name
subfolder_name = 'enhance'

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

# Loop over each image file
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    print("min",np.min(image))
    print("max",np.max(image))
    gaussian_image = convolve(image, gaussian_filter)
    display_save("Gaussian Image", gaussian_image, os.path.join(output_folder, subfolder_name, f"{os.path.splitext(image_file)[0]}_gaussian.jpg"))
    
    residual_image = image - gaussian_image
    display_save("Residual Image", residual_image, os.path.join(output_folder, subfolder_name, f"{os.path.splitext(image_file)[0]}_residual.jpg"))

    enhanced_image = image + residual_image
    display_save("Enhanced Image", enhanced_image, os.path.join(output_folder, subfolder_name, f"{os.path.splitext(image_file)[0]}_sharpen.jpg"))
