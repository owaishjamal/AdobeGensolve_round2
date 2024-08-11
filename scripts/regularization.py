# from utilities import fetch_csv_data, render_plot
import numpy as np
from scipy.spatial import ConvexHull

def classify_shapes(shape_data):
    detected_shapes = []
    for shape in shape_data:
        shape = np.array(shape)  # Convert shape to NumPy array for processing
        if is_circle(shape):
            detected_shapes.append('Circle')
        elif is_ellipse(shape):
            detected_shapes.append('Ellipse')
        elif is_rectangle(shape):
            detected_shapes.append('Rectangle')
        elif is_polygon(shape):
            detected_shapes.append('Polygon')
        elif is_star(shape):
            detected_shapes.append('Star')
        else:
            detected_shapes.append('Unknown')
    return detected_shapes

def is_circle(shape_points):
    center = np.mean(shape_points, axis=0)
    distances = np.linalg.norm(shape_points - center, axis=1)
    return np.std(distances) < 0.05 * np.mean(distances)

def is_ellipse(shape_points):
    covariance_matrix = np.cov(shape_points.T)
    eigenvalues, _ = np.linalg.eigh(covariance_matrix)
    return eigenvalues[0] / eigenvalues[1] > 0.5

def is_rectangle(shape_points):
    convex_hull = ConvexHull(shape_points)
    return len(convex_hull.vertices) == 4

def is_polygon(shape_points):
    convex_hull = ConvexHull(shape_points)
    return len(convex_hull.vertices) > 4

def is_star(shape_points):
    mean_dist = np.mean(np.linalg.norm(shape_points - np.mean(shape_points, axis=0), axis=1))
    max_dist = np.max(np.linalg.norm(shape_points - np.mean(shape_points, axis=0), axis=1))
    return len(shape_points) > 10 and mean_dist > 0.1 * max_dist

# Example usage (adjust paths and calls as necessary)
# shape_data = fetch_csv_data('/content/drive/MyDrive/curve_project/data/isolated.csv')
# shapes = classify_shapes(shape_data)
# print(shapes)
# render_plot(shape_data, title='Detected Shapes')
