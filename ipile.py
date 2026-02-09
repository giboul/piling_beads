import pyvista as pv
from resting_bead import resting_bead
import traceback


state = dict(radius=1, num_spheres=0, spheres=[], selected_spheres=[])

def add_sphere(radius, center, mesh_kwargs=dict(), sphere_kwargs=dict()):
    state["spheres"].append((radius, *center))
    mesh = pv.Sphere(radius=radius, center=center, **sphere_kwargs)
    mesh._id = state["num_spheres"]
    state["num_spheres"] += 1
    pl.add_mesh(mesh, **mesh_kwargs)

def update_radius(value) -> None:
    state["radius"] = round(value, 0)

def mesh_pick(mesh):
    state["selected_spheres"] += state["spheres"][mesh._id]
    if len(state["selected_spheres"])//4 == 3:
        try:
            center = resting_bead(state["radius"], *state["selected_spheres"])
            add_sphere(state["radius"], center, mesh_kwargs=dict(color="red"))
        except ValueError as e:
            traceback.print_tb(e.__traceback__)
            print("ERROR: Solution cannot be computed.", )
        finally:
            state["selected_spheres"].clear()

pl = pv.Plotter()
add_sphere(radius=1, center=(0, 0, 0))
add_sphere(radius=1, center=(2, 0, 0))
add_sphere(radius=1, center=(2, 2, 0))
pl.add_slider_widget(update_radius, [1, 5], title="Radius")
pl.enable_mesh_picking(mesh_pick, left_clicking=True)
pl.show_axes()
pl.show()