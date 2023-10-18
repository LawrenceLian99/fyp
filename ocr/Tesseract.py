import cv2
import numpy as np
import pytesseract

# Open an image using OpenCV with the correct color channel order
image_path = r"C:\Users\USER\OneDrive\Desktop\Lawrence's Assignment\fyp\test_1.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Specify cv2.IMREAD_GRAYSCALE to read as grayscale

# Define the border size (e.g., 10 pixels)
border_size = 10

# Increase the contrast using OpenCV
contrast_factor = 1.75  # You can adjust this value as needed
img_contrast = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)

# Binarize the grayscale image using a lower threshold (e.g., 150)
threshold_value = 250  # Adjust this threshold as needed
_, img_binary = cv2.threshold(img_contrast, threshold_value, 255, cv2.THRESH_BINARY)

# Reverse the binary image to correct the display
img_binary = cv2.bitwise_not(img_binary)

# Rotate the image 5 degrees counterclockwise
angle = 1.5  # Positive angle for counterclockwise rotation
rows, cols = img_binary.shape
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
img_rotated = cv2.warpAffine(img_binary, M, (cols, rows))

# Add a white border around the image
img_with_border = np.ones((rows + 2 * border_size, cols + 2 * border_size), dtype=np.uint8) * 255
img_with_border[border_size:border_size + rows, border_size:border_size + cols] = img_rotated

# Perform OCR using Tesseract on the rotated binary image without PSM configuration
text = pytesseract.image_to_string(img_with_border, config=(
    # only a set of characters
    ' -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' +

    # no language model
    ' -c load_system_dawg=0' +
    ' -c load_freq_dawg=0' +

    ' -c enable_new_segsearch=1' +

    ' -c language_model_penalty_non_freq_dict_word=1' +
    ' -c language_model_penalty_non_dict_word=1'
))

# Replace numbers with characters in the first four characters
first_part = text[:4].replace('0', 'O').replace('1', 'I').replace('2', 'Z')  # Add more replacements as needed

# Replace "2261" with "22G1" in the extracted text
text = text.replace("2261", "22G1")
text = text.replace("4261", "42G1")

# Remove all spaces from the first line of the text
lines = text.split('\n')


if lines:
    # Remove spaces from the first line
    first_line = lines[0].replace(' ', '')

    # Perform replacements within the first four characters of the first line
    first_part = first_line[:4].replace('0', 'O').replace('1', 'I').replace('2', 'Z')  # Add more replacements as needed

    # Reassemble the text with the modified first line
    modified_text = first_part + first_line[4:] + '\n' + '\n'.join(lines[1:])
    print(modified_text)
else:
    print("No text found.")




# Display the image with the white border using OpenCV
cv2.imshow('Image with Border', img_with_border)
cv2.waitKey(0)
cv2.destroyAllWindows()
