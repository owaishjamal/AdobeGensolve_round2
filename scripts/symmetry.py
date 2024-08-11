import numpy as np
# from utilities import fetch_csv_data, render_plot

def analyze_symmetry(shape_data):
    symmetry_results = []
    for shape in shape_data:
        if check_reflection_symmetry(shape):
            symmetry_results.append('Reflection Symmetry')
        elif check_rotational_symmetry(shape):
            symmetry_results.append('Rotational Symmetry')
        else:
            symmetry_results.append('No Symmetry')
    return symmetry_results

def check_reflection_symmetry(points):
    for angle in np.linspace(0, np.pi, 180):
        rotated_points = rotate_points(points, angle)
        if np.allclose(rotated_points, np.flip(rotated_points, axis=0)):
            return True
    return False

def check_rotational_symmetry(points):
    center = np.mean(points, axis=0)
    for angle in np.linspace(0, 2 * np.pi, 360):
        rotated_points = rotate_points(points - center, angle) + center
        if np.allclose(points, rotated_points):
            return True
    return False

def rotate_points(points, angle):
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return points @ rotation_matrix

# if __name__ == "__main__":
#     shape_data = fetch_csv_data('/content/drive/MyDrive/curve_project/data/frag0.csv')
#     symmetries = analyze_symmetry(shape_data)
#     print(symmetries)
#     render_plot(shape_data, title='Symmetry Analysis')
