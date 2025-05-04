import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Initialize global variables for drawing
start_point = None
end_point = None
drawing = False

# Open file dialog to select an image
root = tk.Tk()
root.withdraw()  # Hide the Tkinter window
file_path =  r"D:\Proiect 1-BOS\Frames\schlieren_clasicfiji5.png"

# Load the image
image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale for intensity analysis
if image is None:
    print("Error: Could not load image.")
    exit()

# Function to handle mouse events
def draw_line(event, x, y, flags, param):
    global start_point, end_point, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        drawing = True

    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        drawing = False

        # Draw the selected line
        cv2.line(image_display, start_point, end_point, (255, 255, 255), 2)
        cv2.imshow("Draw Line", image_display)

        # Extract pixel intensities along the line and plot
        extract_and_plot_intensity(image, start_point, end_point)

# Function to extract and plot pixel intensities along the line
def extract_and_plot_intensity(image, start, end):
    # Generate coordinates for the line
    num_points = int(np.linalg.norm(np.array(end) - np.array(start)))  # Estimate number of points
    line_coords_x = np.linspace(start[0], end[0], num=num_points, dtype=int)
    line_coords_y = np.linspace(start[1], end[1], num=num_points, dtype=int)

    # Get pixel intensities
    intensities = [image[y, x] for x, y in zip(line_coords_x, line_coords_y)]

    # Apply smoothing (Moving Average)
    def moving_average(data, window_size=5):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    smoothed_intensities = moving_average(intensities, window_size=5)

    # Plot intensity profile
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(intensities)), intensities, marker="o", linestyle="-", color="gray", label="Original")
    plt.plot(range(len(smoothed_intensities)), smoothed_intensities, marker="o", linestyle="-", color="b", label="Smoothed")
    plt.xlabel("Pixel Position Along Line")
    plt.ylabel("Intensity Value (0-255)")
    plt.title("Smoothed Pixel Intensity Profile")
    plt.legend()
    plt.grid(True)
    plt.show()

# Make a copy of the image for displaying
image_display = image.copy()

# Create OpenCV window and set mouse callback
cv2.namedWindow("Draw Line")
cv2.setMouseCallback("Draw Line", draw_line)

cv2.imshow("Draw Line", image_display)
cv2.waitKey(0)
cv2.destroyAllWindows()