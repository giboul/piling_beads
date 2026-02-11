import numpy as np
import pyvista as pv
from resting_bead import resting_bead
import traceback

class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, value):
        self[key] = value
    def __repr__(self):
        return "<--AttrDict\n"+"\n".join([f"{k}: {v}" for k, v in self.items()])+"\n-->"

state = AttrDict(
    r = 1.,
    ids = np.zeros((0,), dtype=np.int64),
    spheres = np.zeros((0, 4), dtype=np.float64),
    selected_mesh = None,
    selected_spheres = set(),
    mode = "select"
)
print(state)

def update_radius(value) -> None:
    state.radius = value

def add_sphere(radius, center, mesh_kwargs=dict(), sphere_kwargs=dict()):
    if not check_conflicts(radius, center):
        raise ValueError("There is a conflict with another sphere !")
    state.spheres = np.vstack((state.spheres, (radius, *center)))
    mesh = pv.Sphere(radius=radius, center=center, **sphere_kwargs)
    ix = state.ids.max()+1 if state.ids.size else 0
    state.ids = np.hstack((state.ids, ix), dtype=np.int64)
    mesh._id = state.ids.size - 1
    mesh._actor = pl.add_mesh(mesh, pickable=True, **mesh_kwargs)

def check_conflicts(r, xyz, tol=1e-5):
    radii = state.spheres[:, 0]
    centers = state.spheres[:, 1:]
    dist2 = ((centers - xyz)**2).sum(axis=1)
    conflict = (dist2 < (radii+r-tol)**2)
    return not conflict.any()

def drop_sphere():
        try:
            spheres_data = [p for s in state.selected_spheres for p in state.spheres[s]]
            center = resting_bead(state.radius, *spheres_data)
            add_sphere(state.radius, center, mesh_kwargs=dict(color="red"))
        except ValueError as e:
            traceback.print_tb(e.__traceback__)
            print("ERROR: Solution cannot be computed.")
        finally:
            state.selected_spheres.clear()

def delete_sphere():
    mesh = state.selected_mesh
    ix = state.ids[mesh._id]
    print(f"{state.ids = }, {mesh._id = }, {ix = }")
    print("#"*10)
    print(state)
    state.spheres = np.vstack([state.spheres[:ix], state.spheres[ix+1:]])
    print(state.spheres)
    # state.ids = state.ids[state.ids != mesh._id]
    a = np.arange(state.ids.size)
    state.ids[a == mesh._id] = -1
    state.ids[a > mesh._id] -= 1
    print(f"{state.ids = }, {mesh._id = }, {ix = }")
    print("/"*10 + "\n")
    pl.remove_actor(mesh._actor)
    pl.render()
    state.mode = "select"
    state.selected_mesh = None
    state.selected_spheres.clear()

def clear_selection():
    print("Clearing")
    state.selected_spheres.clear()

def mesh_pick(mesh):
    if not hasattr(mesh, "_id"):
        return
    if state.mode == "delete":
        return delete_sphere(mesh)
    state.selected_mesh = mesh
    state.selected_spheres.add(state.ids[mesh._id])
    if len(state.selected_spheres) == 3:
        drop_sphere()

def add_bed_sphere(point):
    point[2] = state.radius
    add_sphere(state.radius, point)

def toggle_bed_mode():
    pl.disable_picking()
    pl.enable_surface_point_picking(add_bed_sphere, left_clicking=True)
    state.mode = "bed"

def toggle_select_mode():
    pl.disable_picking()
    pl.enable_mesh_picking(mesh_pick, left_clicking=True)
    state.mode = "select"

x = np.linspace(-10, 10, num=1_000)
y = np.linspace(-10, 10, num=1_000)
x, y = np.meshgrid(x, y)
z = np.zeros_like(x)
pl = pv.Plotter()
pl.add_mesh(pv.StructuredGrid(x, y, z), opacity=0.5)
add_sphere(radius=1, center=(0, 0, 1))
add_sphere(radius=1, center=(2, 0, 1))
add_sphere(radius=1, center=(0, 2, 1))
add_sphere(radius=1, center=(-2, 0, 1))
add_sphere(radius=1, center=(0, -2, 1))
add_sphere(radius=1, center=(0, -4, 1))
add_sphere(radius=1, center=(-4, 0, 1))
add_sphere(radius=1, center=(-2, -2, 1))
add_sphere(radius=1, center=(2, 2, 1))
pl.add_slider_widget(update_radius, [1, 5], title="Radius")
pl.add_key_event("d", delete_sphere)
pl.add_key_event("b", toggle_bed_mode)
pl.add_key_event("s", toggle_select_mode)
pl.add_key_event("Escape", clear_selection)
toggle_select_mode()
pl.show_axes()
pl.show()