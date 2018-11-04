import numpy as np
import pandas as pd
import os

def create_window(image, x_normalized, y_normalized, half_window_side_normalized,
                  down_sample_width=30, down_sample_height=30):
    x_top_left = (x_normalized - half_window_side_normalized) * image.size[0]
    y_top_left = (y_normalized - half_window_side_normalized) * image.size[1]
    x_bottom_right = (x_normalized + half_window_side_normalized) * image.size[0]
    y_bottom_right = (y_normalized + half_window_side_normalized) * image.size[1]
    box = map(int, (x_top_left, y_top_left, x_bottom_right, y_bottom_right))
    return box, image.crop(box).resize((down_sample_width, down_sample_height))

def design_matrix(images):
    return np.array([np.array(image).flatten() for image in images])

def normalized_phone_coordinates(df, image_file_name):
    image_df = df[df['image'] == image_file_name]
    x_normalized = float(image_df['x'])
    y_normalized = float(image_df['y'])
    return x_normalized, y_normalized


def open_table(folder):
    return pd.read_table(folder + '/labels.txt',
                         sep=' ',
                         names=['image', 'x', 'y'])

def basename(path_string):
    return os.path.basename(os.path.normpath(path_string))
