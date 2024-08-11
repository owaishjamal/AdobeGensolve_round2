import numpy as np
from scipy.interpolate import splprep, splev
# from utilities import fetch_csv_data, render_plot

def finalize_curves(shape_data):
    refined_curves = []
    for curve in shape_data:
        refined_curve = fill_missing_segments(curve)
        refined_curves.append(refined_curve)
    return refined_curves

def fill_missing_segments(curve_points):
    tck, param = splprep([curve_points[:, 0], curve_points[:, 1]], s=0)
    param_new = np.linspace(param.min(), param.max(), len(curve_points))
    x_new, y_new = splev(param_new, tck, der=0)
    return np.vstack((x_new, y_new)).T

# if __name__ == "__main__":
#     shape_data = fetch_csv_data('/content/drive/MyDrive/curve_project/data/occlusion1.csv')
#     refined_curves = finalize_curves(shape_data)
#     render_plot(refined_curves, title='Refined Curves')
