"""
The `Registration` class: rigid alignment
================================================================

This notebook shows how to align two triangle meshes that are not in correspondence

- Losses: NearestNeighbors, LandmarkLoss, sum
- Model: rigid motion
"""

# %% [markdown]
# Load and preprocess data
# -------------------------
#
# Load two triangle meshes from pyvista examples and rescale them to fit in the
# unit box.

# %%
import pykeops
import pyvista as pv
import torch
from pyvista import examples

import skshapes as sks

color_1 = 'tan'
color_2 = 'brown'

# shape1 = sks.PolyData(examples.download_human())
shape1 = sks.PolyData(examples.download_woman().rotate_y(90))
shape2 = sks.PolyData(examples.download_doorman())
shape1.point_data.clear()
shape2.point_data.clear()

def bounds(shape):
    return torch.max(shape.points, dim=0).values - torch.min(shape.points, dim=0).values

lims1 = bounds(shape1)
lims2 = bounds(shape2)
rescale1 = torch.max(lims1)
shape1.points -= torch.min(shape1.points, dim=0).values
shape1.points /= rescale1

rescale2 = torch.max(lims2)
shape2.points -= torch.min(shape2.points, dim=0).values
shape2.points /= rescale2

plotter = pv.Plotter()
plotter.add_mesh(shape1.to_pyvista(), color=color_1, **kwargs)
plotter.add_mesh(shape2.to_pyvista(), color=color_2, **kwargs)

plotter.show()


# %% [markdown]
# Run rigid registration
# ----------------------
#
# with `NearestNeighborsLoss`.

# %%

loss = sks.NearestNeighborsLoss()
# The parameter n_steps is the number of steps for the motion. For a rigid
# motion, it has no impact on the result as the motion is fully determined by
# a rotation matrix and a translation vector. It is however useful for
# creating a smooth animation of the registration.
model = sks.RigidMotion(n_steps=5)

registration = sks.Registration(
    model=model,
    loss=loss,
    n_iter=2,
    verbose=True,
)

registration.fit(source=shape2, target=shape1)
morph = registration.transform(source=shape2)

plotter = pv.Plotter()
plotter.add_mesh(shape1.to_pyvista(), color=color_1, **kwargs)
plotter.add_mesh(morph.to_pyvista(), color=color_2, **kwargs)
plotter.show()

# %% [markdown]
# Add landmarks
# -------------
#
#  in order to indicate left/right, up/down.

# %%
if not pv.BUILDING_GALLERY:
    # If not in the gallery, we can use vedo to open the landmark setter
    # Setting the default backend to vtk is necessary when running in a notebook
    import vedo
    vedo.settings.default_backend= 'vtk'
    sks.LandmarkSetter([shape1, shape2]).start()
else:
    # Set the landmarks manually
    landmarks1 = [4808, 147742, 1774]
    landmarks2 = [325, 2116, 1927]

    shape1.landmark_indices = landmarks1
    shape2.landmark_indices = landmarks2

colors = ["red", "green", "blue"]
plotter = pv.Plotter()
plotter.add_mesh(shape1.to_pyvista(), color=color_1, **kwargs)
for i in range(len(shape1.landmark_indices)):
    plotter.add_points(
        shape1.landmark_points[i].numpy(),
        color=colors[i % 3],
        render_points_as_spheres=True,
        point_size=25,
    )
plotter.add_mesh(shape2.to_pyvista(), color=color_2, **kwargs)
for i in range(len(shape2.landmark_indices)):
    plotter.add_points(
        shape2.landmark_points[i].numpy(),
        color=colors[i % 3],
        render_points_as_spheres=True,
        point_size=25,
    )
plotter.show()


# %% [markdown]
# Register again
# --------------
#
# with loss = `NearestNeighborsLoss` + `LandmarkLoss`

# %%

loss_landmarks = sks.NearestNeighborsLoss() + sks.LandmarkLoss()

registration = sks.Registration(
    model=model,
    loss=loss_landmarks,
    n_iter=2,
    verbose=True,
)

registration.fit(source=shape2, target=shape1)
morph = registration.transform(source=shape2)

plotter = pv.Plotter()
plotter.add_mesh(shape1.to_pyvista(), color=color_1, **kwargs)
plotter.add_mesh(morph.to_pyvista(), color=color_2, **kwargs)
plotter.show()
