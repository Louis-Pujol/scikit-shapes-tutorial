"""
The `Multiscale` class: representing a shape at different scales
================================================================

Text
"""

# %%
from pyvista import examples
import pyvista as pv
import skshapes as sks

cpos = [(0.07817110755919496, 0.13558926405422117, 0.5210700195677971),
 (-0.01684039831161499, 0.11015420686453581, -0.0015369504690170288),
 (-0.26589633341798463, 0.9640005779198042, 0.0014232516134205578)]

bunny = sks.PolyData(examples.download_bunny())

bunny.plot(cpos=cpos)

# %%
if not pv.BUILDING_GALLERY:
    # If not in the gallery, we can use vedo to open the landmark setter
    # Setting the default backend to vtk is necessary when running in a notebook
    import vedo
    vedo.settings.default_backend= 'vtk'
    sks.LandmarkSetter(bunny).start()
else:
    # Set the landmarks manually
    bunny.landmark_indices = [ 4695, 12902,  1368, 13223,  5460, 20809, 12829, 10080, 30769, 27127, 21764, 23356]

# Add landmarks
# sks.LandmarkSetter(bunny).start()

# Add signal
bunny.point_data["height"] = bunny.points[:, 1]

# Add control points
bunny.control_points = bunny.bounding_grid(N=10, offset=0.05)

# Plot
plotter = pv.Plotter()
plotter.add_mesh(bunny.to_pyvista(), scalars="height")
plotter.add_points(bunny.landmark_points.numpy(), color="red", point_size=10, render_points_as_spheres=True)
plotter.add_mesh(bunny.control_points.to_pyvista(), color="green", opacity=0.9)
plotter.camera_position = cpos
plotter.show()

# %%
ratios = [0.5, 0.1, .01]
multimesh = sks.Multiscale(bunny, ratios=ratios)

plotter = pv.Plotter(shape=(1, 4))
for i, ratio in enumerate([1] + ratios):
    plotter.subplot(0, i)
    plotter.add_mesh(multimesh.at(ratio=ratio).to_pyvista(), color="tan")
    plotter.add_text(f"Ratio: {ratio}\n\n{multimesh.at(ratio=ratio).n_points} points", font_size=16)
    plotter.camera_position = cpos

plotter.show()

# %%
policy = sks.FineToCoarsePolicy(reduce="mean")
multimesh.propagate(signal_name="height", from_ratio=1)

for ratio in ratios:
    multimesh.at(ratio=ratio).control_points = multimesh.at(ratio=1).control_points

plotter = pv.Plotter(shape=(1, 4))
for i, ratio in enumerate([1] + ratios):
    plotter.subplot(0, i)
    plotter.add_mesh(multimesh.at(ratio=ratio).to_pyvista(), scalars="height")
    plotter.add_points(multimesh.at(ratio=ratio).landmark_points.numpy(), color="red", point_size=10, render_points_as_spheres=True)
    plotter.add_mesh(multimesh.at(ratio=ratio).control_points.to_pyvista(), color="green", opacity=0.9)
    plotter.add_text(f"Ratio: {ratio}", font_size=24)
    plotter.camera_position = cpos

plotter.show()
