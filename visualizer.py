import pyvista as pv
import numpy as np
import colorcet as cc
from matplotlib import colors

# Testing data
ERROR_MIN = 0.0
ERROR_MAX = 3

norm = colors.Normalize(vmin=ERROR_MIN, vmax=ERROR_MAX)
cmap = colors.ListedColormap(cc.cm.fire)

references = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

true_point = np.array([0.5, 0.5, 0.5])
estimated_point = np.array([0.5, 0.5, 0.0])


pl = pv.Plotter()
pl.show_grid()

def add_points(point, color, point_size, render_points_as_spheres):
    pl.add_points(
        point,
        color = color,
        point_size = point_size,
        render_points_as_spheres = render_points_as_spheres
    )

def plot_error(point, error):
    u = norm(error)
    rgba = cmap(u)
    color = tuple(rgba[:3])
    add_points(point, color, 10, False)

add_points(references, 'black', 20, True)

#Temporary error visualization testing
for i in range(10):
    for j in range(10):
        for k in range(10):
            point = np.array([i/10, j/10, k/10])
            error = i+j+k
            plot_error(point, error/30)


#Final Setups
pl.camera_position = [
    (4, 2, 2.5),
    (0.5, 0.5, 0.5),    # focal point
    (0, 0, 1)
]

pl.show_bounds(
    grid='back',
    all_edges=True,
    location='outer',
    bounds=(0,1,0,1,0,1)
)

pl.add_scalar_bar(
    title="Error (m)",
    vertical=True,
    n_labels=5,
    fmt="%.3f"
)

pl.show()