from PIL import Image
import cv2
import numpy as np
import os
import math

# Directory where you want to save the image
# save_directory = 'blured_mask'
# eroded_directory = 'eroded_mask'
# edges_directory = 'edges_mask'
output_folder = 'final_trans'



# Create the directory if it doesn't exist
# if not os.path.exists(save_directory):
#     os.makedirs(save_directory)

# if not os.path.exists(edges_directory):
#     os.makedirs(edges_directory)

# if not os.path.exists(eroded_directory):
#     os.makedirs(eroded_directory)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def angle_between_line_and_vertical_line(m):
    angle_radians = math.atan(m)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def alpha_line(x, y, alpha):

    length=150
    # Convert angle from degrees to radians
    alpha_rad = np.radians(alpha)

    # Calculate direction vector (dx, dy)
    # Since alpha is with respect to vertical, we swap sin and cos
    dx = np.sin(alpha_rad)
    dy = np.cos(alpha_rad)

    # Calculate end point of the line
    x_end = x + length * dx
    y_end = y + length * dy

    return (x,y), (int(x_end), int(y_end))

def extract_data(image):
    print('✅ here blured erosion edges')
    height, width = image.shape
    # Blur the img
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # blured_path = os.path.join(save_directory, 'll_predict_output.png')
    # cv2.imwrite(blured_path, blurred)

    # Erode the blurred image
    kernel = np.ones((7, 7), np.uint8)
    erosion = cv2.erode(blurred, kernel, iterations=1)
    # eroded_path = os.path.join(eroded_directory, 'll_eroded_output.png')
    # cv2.imwrite(eroded_path, erosion)

    # Detect edges on the eroded image
    edges = cv2.Canny(erosion, 50, 80)

    # Save the edges image
    # edges_filename = os.path.join(edges_directory, 'll_edged_output.png')
    # cv2.imwrite(edges_filename, edges)


    # Apply the Hough Line Transform to detect lines
    print('✅ here: hought transform')

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 30)  # 250

    # Count the number of detected lanes
    if lines is not None:
        num_lanes = len(lines)
        print(f"Number of detected lanes: {num_lanes}")

    # Extract line parameters and draw bold lines
    m_neg_list, n_neg_list = [],  []
    m_pos_list, n_pos_list = [],  []

    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            m, n = rho / np.sin(theta), -1 / np.tan(theta)

            if n < 0:
                m_neg_list.append(m)
                n_neg_list.append(n)
            else:
                m_pos_list.append(m)
                n_pos_list.append(n)
    
        m_neg, n_neg = np.mean(m_neg_list), np.mean(n_neg_list)
        m_pos, n_pos = np.mean(m_pos_list), np.mean(n_pos_list)
        print(f'y = {m_neg:.2f} + {n_neg:.2f}*x')
        print(f'y = {m_pos:.2f} + {n_pos:.2f}*x')

        # Function to calculate y for a given x using the line equation
        def calculate_y(m, n, x):
            res = int(n * x + m)
            return res
        # Function to calculate x for a given y using the line equation

        def calculate_x(m, n, y):
            return int((y - m) / n) if n != 0 else 0

        # Define the y-interval
        y1_interval = height - 100
        y2_interval = height

        x1_neg = calculate_x(m_neg, n_neg, y1_interval)
        x2_neg = calculate_x(m_neg, n_neg, y2_interval)
        x1_pos = calculate_x(m_pos, n_pos, y1_interval)
        x2_pos = calculate_x(m_pos, n_pos, y2_interval)

        cv2.line(color_image, (x1_neg, y1_interval), (x2_neg, y2_interval), (0, 255, 255), 5)  # Negative slope line
        cv2.line(color_image, (x2_neg, y1_interval), (x2_neg, y2_interval), (0, 100, 255), 5)  # Negative vertical line
        cv2.line(color_image, (x1_pos, y1_interval), (x2_pos, y2_interval), (0, 255, 255), 5)  # Positive slope line
        cv2.line(color_image, (x2_pos, y1_interval), (x2_pos, y2_interval), (0, 100, 255), 5)  # Positive vertical line

        # Draw line of direction 
        alpha_neg = angle_between_line_and_vertical_line(n_neg)
        alpha_pos = angle_between_line_and_vertical_line(n_pos)
        alpha = (alpha_neg + alpha_pos) / 2
        print('▶️ alpha:',alpha)
        x1_alpha = (x2_neg + x2_pos)//2
        y1_alpha = height
        x2_alpha, y2_alpha = alpha_line(x1_alpha, y1_alpha, alpha)[1]
        cv2.line(color_image, (x1_alpha, y1_alpha), (x2_alpha, y1_alpha - abs(y2_alpha - y1_alpha)), (255, 0, 0), 5)  # alpha line
        cv2.line(color_image, (x1_alpha, y1_alpha), (x1_alpha, y1_alpha - (y2_alpha - y1_alpha)), (255, 0, 100), 5)  # vertical line
        
        output_filename = os.path.join(output_folder, 'final_output.png')
        cv2.imwrite(output_filename, color_image)

    else:
        print("▶️ Line is  None")