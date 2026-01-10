import pyvista as pv
import numpy as np
import colorcet as cc
from matplotlib import colors
import cmake_mlat

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
ERROR_MAX = 6.377745716588144e-16

norm = colors.Normalize(vmin=ERROR_MIN, vmax=ERROR_MAX)
cmap = cc.cm.fire_r

references = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

pl = pv.Plotter()
pl.show_grid()

def add_points(point, color, point_size, render_points_as_spheres):
    pl.add_points(
        point,
        color = color,
        point_size = point_size,
        render_points_as_spheres = render_points_as_spheres
    )
add_points(references, 'black', 25, True)

points = []
errors = []

print("start")
for i in range(6):
    for j in range(6):
        for k in range(6):
            true_point = np.array([i/5, j/5, k/5], dtype=np.float64)
            ranges = cmake_mlat.find_ranges(true_point, references)
            estimated_point = cmake_mlat.find_point(references, ranges)
            scalar_error = cmake_mlat.find_scalar_error(true_point, estimated_point)
            field.add_sample(true_point, scalar_error)
            print(scalar_error)
print("end")


cloud = field.to_polydata()

pl.add_mesh(
    cloud,
    scalars="error",
    cmap=cc.cm.fire_r,
    clim=(ERROR_MIN, ERROR_MAX),
    point_size=35,
    render_points_as_spheres= False
)
pl.remove_scalar_bar()

scalar_bar = pl.add_scalar_bar(
    title="Scalar Error",
    vertical= True,
    fmt="%.2e",
    n_labels=6
)

scalar_bar.SetPosition(0.85, 0.1)

#Final Setups
pl.camera_position = [
    (4, 2.5, 2.25),
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
