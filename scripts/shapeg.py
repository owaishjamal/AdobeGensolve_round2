import numpy as np
import csv

def create_polyline(shape_name, num_points=50):
    if shape_name == 'circle':
        angles = np.linspace(0, 2 * np.pi, num_points)
        radius = 10  # fixed radius
        x_coords = radius * np.cos(angles)
        y_coords = radius * np.sin(angles)
    elif shape_name == 'ellipse':
        angles = np.linspace(0, 2 * np.pi, num_points)
        major_axis, minor_axis = 10, 5  # semi-major and semi-minor axes
        x_coords = major_axis * np.cos(angles)
        y_coords = minor_axis * np.sin(angles)
    elif shape_name == 'rectangle':
        x_coords = np.array([0, 10, 10, 0, 0])
        y_coords = np.array([0, 0, 5, 5, 0])
    elif shape_name == 'polygon':
        sides = 6  # hexagon
        angles = np.linspace(0, 2 * np.pi, sides + 1)
        x_coords = 10 * np.cos(angles)
        y_coords = 10 * np.sin(angles)
    elif shape_name == 'star':
        angles = np.linspace(0, 2 * np.pi, num_points)
        radii = np.where(np.arange(num_points) % 2 == 0, 10, 5)
        x_coords = radii * np.cos(angles)
        y_coords = radii * np.sin(angles)
    else:
        raise ValueError("Unsupported shape_name. Use 'circle', 'ellipse', 'rectangle', 'polygon', or 'star'.")

    return np.stack([x_coords, y_coords], axis=-1)

def export_polyline_to_csv(polyline_data, file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "X", "Y"])  # Header
        for idx, (x, y) in enumerate(polyline_data):
            writer.writerow([1, x, y])  # ID is kept constant for simplicity

# Generate and save polylines for various shapes
shape_types = ['circle', 'ellipse', 'rectangle', 'polygon', 'star']
for shape in shape_types:
    polyline_data = create_polyline(shape)
    export_polyline_to_csv(polyline_data, f'/content/drive/MyDrive/curve_project/data/{shape}.csv')
