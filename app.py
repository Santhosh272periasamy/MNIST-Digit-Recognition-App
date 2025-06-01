import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from streamlit_drawable_canvas import st_canvas

import warnings

warnings.filterwarnings('ignore')

 # Initialize a global list to store predicted digits
dgrs = []
# Initialize a global string to store the prediction result
res = " "

def predict():
    global res
    model = load_model('mnist.h5') # Load the pre-trained MNIST model (expects input images of size 28x28)

    image_folder = "./"  # Define path to the image saved from the canvas
    filename = f'img.jpg'

    # Working on the captured image to match the model input
    # Read the image in color format using OpenCV
    image = cv2.imread(image_folder + filename, cv2.IMREAD_COLOR)

    # Convert the image to grayscale to simplify the image data (removes color channels)
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth the image and reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to convert the grayscale image to binary (black & white)
     # Adaptive thresholding is better than global thresholding in varying lighting conditions

    th = cv2.adaptiveThreshold( blurred,                   # Source image

        255,                          # Maximum value to use with THRESH_BINARY
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Use a weighted sum of neighborhood values
        cv2.THRESH_BINARY_INV,        # Invert the output (digits become white)
        11, 2                             # Block size and constant C
    )


     # Find contours (continuous lines or curves that bound the white regions)

    contours = cv2.findContours(
        th,                                                           # Binary image
        cv2.RETR_EXTERNAL,                 # Only retrieve external contours
        cv2.CHAIN_APPROX_SIMPLE  # Compress horizontal, vertical, and diagonal segments
    )[0]                                                            # Only need the contours details

    # Loop through each contour (likely each digit drawn)

    for cnt in contours:

        # Compute the bounding rectangle around the contour

        x, y, w, h = cv2.boundingRect(cnt)

     # Draw a blue rectangle around each detected digit

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
    # Crop the digit from the binary image using the bounding box

        digit = th[y:y + h, x:x + w]
    # Resize the digit to 18x18 pixels (MNIST model expects 28x28)

        resized_digit = cv2.resize(digit, (18, 18))



     # Pad the resized digit with 5 pixels of black pixels (zeros) on each side

    # This results in a 28x28 image as expected by the model

        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

         
    # Reshape the image to match the model's input shape: (1, 28, 28, 1)

        digit = padded_digit.reshape(1, 28, 28, 1)

    # Normalize pixel values from [0, 255] to [0, 1]

        digit = digit / 255.0

    # Get the prediction probabilities for each digit (0–9)

        pred = model.predict(digit)[0]

    # Get the digit with the highest probability (model's final prediction)

        final_pred = np.argmax(pred)

     # Append predicted digit to the global list

        dgrs.append(int(final_pred))

 

    # Add predicted digit to the result string

        res = res + " " + str(final_pred)

 

    # Prepare text showing prediction and confidence percentage

        data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'
    
     # Define font settings for overlaying prediction on the image

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 255, 255)  # White color
        thickness = 1
    
    # Overlay prediction text on the image at the top-left corner of the bounding box

        cv2.putText(image, data, (x, y), font, fontScale, color, thickness)



st.title("Drawable Canvas")

st.markdown("""

Draw digits on the canvas, get the image data back into Python!

""")

 #Create a canvas where users can draw digits

canvas_result = st_canvas(
    stroke_width=10,                   # Thickness of the brush
    stroke_color='red',                # Color of the brush
    height=150                               # Height of the canvas
) 

# Check if the user has drawn something
if canvas_result.image_data is not None:

    # Save the drawn image to a file for processing

    cv2.imwrite(f"img.jpg",  canvas_result.image_data)

# Create a "Predict" button

if st.button("Predict"):
    predict()                                                   # Call the predict function
    st.write('The predicted digit:', res)  # Display the result on the app