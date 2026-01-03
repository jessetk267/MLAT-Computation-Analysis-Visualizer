import pyvista as pv
import numpy as np
import colorcet as cc
from matplotlib import colors

class ErrorField:
    def __init__(self):
        self.points = []
        self.errors = []

    def add_sample(self, point, error):
        self.points.append(point)
        self.errors.append(error)

    def to_polydata(self):
        cloud = pv.PolyData(np.array(self.points))
        cloud["error"] = np.array(self.errors)
        return cloud

field = ErrorField()
ERROR_MIN = 0.0
ERROR_MAX = 3

norm = colors.Normalize(vmin=ERROR_MIN, vmax=ERROR_MAX)
cmap = cc.cm.fire

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
add_points(references, 'black', 20, True)

points = []
errors = []

#Temporary error visualization testing
for i in range(5):
    for j in range(5):
        for k in range(5):
            point = np.array([i/5, j/5, k/5])
            error = (i+j+k)/5
            field.add_sample(point, error)


cloud = field.to_polydata()

pl.add_mesh(
    cloud,
    scalars="error",
    cmap=cc.cm.fire_r,
    clim=(ERROR_MIN, ERROR_MAX),
    point_size=15,
    render_points_as_spheres= False
)
pl.remove_scalar_bar()

scalar_bar = pl.add_scalar_bar(
    title="Error",
    vertical= True,
    fmt="%.4f",
    n_labels=6
)

scalar_bar.SetPosition(0.85, 0.1)

#Final Setups
pl.camera_position = [
    (4, 1.5, 2),
    (0.5, 0.5, 0.5),    # focal point
    (0, 0, 1)
]

pl.show_bounds(
    grid='back',
    all_edges=True,
    location='outer',
    bounds=(0,1,0,1,0,1)
)

pl.show()