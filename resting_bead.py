"""
Module for finding the position of a single sphere of known radius 
in tangent contact with three other spheres.

Use the `resting_bead` function for functionnality or 
`naive_resting_bead` for direct but unsteady calculation.
"""
import numpy as np


def naive_resting_bead(r, r_1, x_1, y_1, z_1, r_2, x_2, y_2, z_2, r_3, x_3, y_3, z_3, s=+1):
    """
    From the radius of the new sphere and the coordinates and radii of three other spheres,
    find the position of the tangent fourth sphere. This solution is not valid for cases where 
    x_1 == x_2 or the centers are on a vertical plane (xz or yz).
    
    :param r: Radius of the new sphere
    :param r_1: Radius of the first sphere
    :param x_1: `x` coordinate of the first sphere
    :param y_1: `y` coordinate of the first sphere
    :param z_1: `z` coordinate of the first sphere
    :param r_2: Radius of the second sphere
    :param x_2: `x` coordinate of the second sphere
    :param y_2: `y` coordinate of the second sphere
    :param z_2: `z` coordinate of the second sphere
    :param r_3: Radius of the third sphere
    :param x_3: `x` coordinate of the third sphere
    :param y_3: `y` coordinate of the third sphere
    :param z_3: `z` coordinate of the third sphere
    :param s: Which solution to keep: s=+1 top solution, s=-1 -> bottom solution
    """
    z =(s*(((-2*z_1*(-x_1*y_2 + x_1*y_3 + x_2*y_1 - x_2*y_3 - x_3*y_1 + x_3*y_2)**2 + (x_1*z_2 - x_1*z_3 - x_2*z_1 + x_2*z_3 + x_3*z_1 - x_3*z_2)*(-2*r*r_1*x_2 + 2*r*r_1*x_3 + 2*r*r_2*x_1 - 2*r*r_2*x_3 - 2*r*r_3*x_1 + 2*r*r_3*x_2 - r_1**2*x_2 + r_1**2*x_3 + r_2**2*x_1 - r_2**2*x_3 - r_3**2*x_1 + r_3**2*x_2 + x_1**2*x_2 - x_1**2*x_3 - x_1*x_2**2 + x_1*x_3**2 + 2*x_1*y_1*y_2 - 2*x_1*y_1*y_3 - x_1*y_2**2 + x_1*y_3**2 - x_1*z_2**2 + x_1*z_3**2 + x_2**2*x_3 - x_2*x_3**2 - x_2*y_1**2 + 2*x_2*y_1*y_3 - x_2*y_3**2 + x_2*z_1**2 - x_2*z_3**2 + x_3*y_1**2 - 2*x_3*y_1*y_2 + x_3*y_2**2 - x_3*z_1**2 + x_3*z_2**2) + (-y_1*z_2 + y_1*z_3 + y_2*z_1 - y_2*z_3 - y_3*z_1 + y_3*z_2)*(2*r*r_1*y_2 - 2*r*r_1*y_3 - 2*r*r_2*y_1 + 2*r*r_2*y_3 + 2*r*r_3*y_1 - 2*r*r_3*y_2 + r_1**2*y_2 - r_1**2*y_3 - r_2**2*y_1 + r_2**2*y_3 + r_3**2*y_1 - r_3**2*y_2 + x_1**2*y_2 - x_1**2*y_3 - 2*x_1*x_2*y_1 + 2*x_1*x_2*y_3 + 2*x_1*x_3*y_1 - 2*x_1*x_3*y_2 + x_2**2*y_1 - x_2**2*y_3 - x_3**2*y_1 + x_3**2*y_2 - y_1**2*y_2 + y_1**2*y_3 + y_1*y_2**2 - y_1*y_3**2 + y_1*z_2**2 - y_1*z_3**2 - y_2**2*y_3 + y_2*y_3**2 - y_2*z_1**2 + y_2*z_3**2 + y_3*z_1**2 - y_3*z_2**2))**2 - ((-x_1*y_2 + x_1*y_3 + x_2*y_1 - x_2*y_3 - x_3*y_1 + x_3*y_2)**2 + (x_1*z_2 - x_1*z_3 - x_2*z_1 + x_2*z_3 + x_3*z_1 - x_3*z_2)**2 + (-y_1*z_2 + y_1*z_3 + y_2*z_1 - y_2*z_3 - y_3*z_1 + y_3*z_2)**2)*(4*z_1**2*(-x_1*y_2 + x_1*y_3 + x_2*y_1 - x_2*y_3 - x_3*y_1 + x_3*y_2)**2 - 4*(r + r_1)**2*(-x_1*y_2 + x_1*y_3 + x_2*y_1 - x_2*y_3 - x_3*y_1 + x_3*y_2)**2 + (-2*r*r_1*x_2 + 2*r*r_1*x_3 + 2*r*r_2*x_1 - 2*r*r_2*x_3 - 2*r*r_3*x_1 + 2*r*r_3*x_2 - r_1**2*x_2 + r_1**2*x_3 + r_2**2*x_1 - r_2**2*x_3 - r_3**2*x_1 + r_3**2*x_2 + x_1**2*x_2 - x_1**2*x_3 - x_1*x_2**2 + x_1*x_3**2 + 2*x_1*y_1*y_2 - 2*x_1*y_1*y_3 - x_1*y_2**2 + x_1*y_3**2 - x_1*z_2**2 + x_1*z_3**2 + x_2**2*x_3 - x_2*x_3**2 - x_2*y_1**2 + 2*x_2*y_1*y_3 - x_2*y_3**2 + x_2*z_1**2 - x_2*z_3**2 + x_3*y_1**2 - 2*x_3*y_1*y_2 + x_3*y_2**2 - x_3*z_1**2 + x_3*z_2**2)**2 + (2*r*r_1*y_2 - 2*r*r_1*y_3 - 2*r*r_2*y_1 + 2*r*r_2*y_3 + 2*r*r_3*y_1 - 2*r*r_3*y_2 + r_1**2*y_2 - r_1**2*y_3 - r_2**2*y_1 + r_2**2*y_3 + r_3**2*y_1 - r_3**2*y_2 + x_1**2*y_2 - x_1**2*y_3 - 2*x_1*x_2*y_1 + 2*x_1*x_2*y_3 + 2*x_1*x_3*y_1 - 2*x_1*x_3*y_2 + x_2**2*y_1 - x_2**2*y_3 - x_3**2*y_1 + x_3**2*y_2 - y_1**2*y_2 + y_1**2*y_3 + y_1*y_2**2 - y_1*y_3**2 + y_1*z_2**2 - y_1*z_3**2 - y_2**2*y_3 + y_2*y_3**2 - y_2*z_1**2 + y_2*z_3**2 + y_3*z_1**2 - y_3*z_2**2)**2))/(-x_1*y_2 + x_1*y_3 + x_2*y_1 - x_2*y_3 - x_3*y_1 + x_3*y_2)**4)**0.5*(-x_1*y_2 + x_1*y_3 + x_2*y_1 - x_2*y_3 - x_3*y_1 + x_3*y_2)**2 + 2*z_1*(-x_1*y_2 + x_1*y_3 + x_2*y_1 - x_2*y_3 - x_3*y_1 + x_3*y_2)**2 - (x_1*z_2 - x_1*z_3 - x_2*z_1 + x_2*z_3 + x_3*z_1 - x_3*z_2)*(-2*r*r_1*x_2 + 2*r*r_1*x_3 + 2*r*r_2*x_1 - 2*r*r_2*x_3 - 2*r*r_3*x_1 + 2*r*r_3*x_2 - r_1**2*x_2 + r_1**2*x_3 + r_2**2*x_1 - r_2**2*x_3 - r_3**2*x_1 + r_3**2*x_2 + x_1**2*x_2 - x_1**2*x_3 - x_1*x_2**2 + x_1*x_3**2 + 2*x_1*y_1*y_2 - 2*x_1*y_1*y_3 - x_1*y_2**2 + x_1*y_3**2 - x_1*z_2**2 + x_1*z_3**2 + x_2**2*x_3 - x_2*x_3**2 - x_2*y_1**2 + 2*x_2*y_1*y_3 - x_2*y_3**2 + x_2*z_1**2 - x_2*z_3**2 + x_3*y_1**2 - 2*x_3*y_1*y_2 + x_3*y_2**2 - x_3*z_1**2 + x_3*z_2**2) - (-y_1*z_2 + y_1*z_3 + y_2*z_1 - y_2*z_3 - y_3*z_1 + y_3*z_2)*(2*r*r_1*y_2 - 2*r*r_1*y_3 - 2*r*r_2*y_1 + 2*r*r_2*y_3 + 2*r*r_3*y_1 - 2*r*r_3*y_2 + r_1**2*y_2 - r_1**2*y_3 - r_2**2*y_1 + r_2**2*y_3 + r_3**2*y_1 - r_3**2*y_2 + x_1**2*y_2 - x_1**2*y_3 - 2*x_1*x_2*y_1 + 2*x_1*x_2*y_3 + 2*x_1*x_3*y_1 - 2*x_1*x_3*y_2 + x_2**2*y_1 - x_2**2*y_3 - x_3**2*y_1 + x_3**2*y_2 - y_1**2*y_2 + y_1**2*y_3 + y_1*y_2**2 - y_1*y_3**2 + y_1*z_2**2 - y_1*z_3**2 - y_2**2*y_3 + y_2*y_3**2 - y_2*z_1**2 + y_2*z_3**2 + y_3*z_1**2 - y_3*z_2**2))/(2*((-x_1*y_2 + x_1*y_3 + x_2*y_1 - x_2*y_3 - x_3*y_1 + x_3*y_2)**2 + (x_1*z_2 - x_1*z_3 - x_2*z_1 + x_2*z_3 + x_3*z_1 - x_3*z_2)**2 + (-y_1*z_2 + y_1*z_3 + y_2*z_1 - y_2*z_3 - y_3*z_1 + y_3*z_2)**2)) 
    y = ((-x_1 + x_2)*(-x_1**2 + x_3**2 - y_1**2 + y_3**2 + 2*z*(z_1 - z_3) - z_1**2 + z_3**2 + (r + r_1)**2 - (r + r_3)**2) - (-x_1 + x_3)*(-x_1**2 + x_2**2 - y_1**2 + y_2**2 + 2*z*(z_1 - z_2) - z_1**2 + z_2**2 + (r + r_1)**2 - (r + r_2)**2))/(2*((-x_1 + x_2)*(-y_1 + y_3) - (-x_1 + x_3)*(-y_1 + y_2)))
    x = (-(-y_1 + y_2)*((-x_1 + x_2)*(-x_1**2 + x_3**2 - y_1**2 + y_3**2 + 2*z*(z_1 - z_3) - z_1**2 + z_3**2 + (r + r_1)**2 - (r + r_3)**2) - (-x_1 + x_3)*(-x_1**2 + x_2**2 - y_1**2 + y_2**2 + 2*z*(z_1 - z_2) - z_1**2 + z_2**2 + (r + r_1)**2 - (r + r_2)**2)) + ((-x_1 + x_2)*(-y_1 + y_3) - (-x_1 + x_3)*(-y_1 + y_2))*(-x_1**2 + x_2**2 - y_1**2 + y_2**2 + 2*z*(z_1 - z_2) - z_1**2 + z_2**2 + (r + r_1)**2 - (r + r_2)**2))/(2*(-x_1 + x_2)*((-x_1 + x_2)*(-y_1 + y_3) - (-x_1 + x_3)*(-y_1 + y_2)))
    return x, y, z

def are_aligned(p1, p2, v):
    """Check if vector p2-p1 and v do not point to opposite ways."""
    return np.dot((p2-p1), v) > 0


def swap_beads(r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3):
    """Exchange the first and the thir spheres."""
    return r3, x3, y3, z3, r2, x2, y2, z2, r1, x1, y1, z1

def swap_axes(r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3):
    """Work on a plane where the linear solution exists."""
    if not np.isclose((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1), 0): return "xy", r2, x2, y2, z2, r1, x1, y1, z1, r3, x3, y3, z3
    if not np.isclose((z2-z1)*(y3-y1) - (z3-z1)*(y2-y1), 0): return "zy", r2, z2, y2, x2, r1, z1, y1, x1, r3, z3, y3, x3
    if not np.isclose((x2-x1)*(z3-z1) - (x3-x1)*(z2-z1), 0): return "xz", r2, x2, z2, y2, r1, x1, z1, y1, r3, x3, z3, y3
    raise ValueError(r"¯\_(ツ)_/¯")

def restore_swapped_axes(plane, x, y, z):
    """Restore the original axes from the `swap_axes` transform."""
    if plane == "xy": return x, y, z
    if plane == "xz": return x, z, y
    if plane == "zy": return z, y, x
    raise ValueError(r"¯\_(ツ)_/¯")

def resting_bead(r, r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3, gravity=np.array((0, 0, -1))):
    """
    From the radius of the new sphere and the coordinates and radii of three other spheres,
    find the position of the tangent fourth sphere.

    :param r: Radius of the new sphere
    :param r_1: Radius of the first sphere
    :param x_1: `x` coordinate of the first sphere
    :param y_1: `y` coordinate of the first sphere
    :param z_1: `z` coordinate of the first sphere
    :param r_2: Radius of the second sphere
    :param x_2: `x` coordinate of the second sphere
    :param y_2: `y` coordinate of the second sphere
    :param z_2: `z` coordinate of the second sphere
    :param r_3: Radius of the third sphere
    :param x_3: `x` coordinate of the third sphere
    :param y_3: `y` coordinate of the third sphere
    :param z_3: `z` coordinate of the third sphere
    :param gravity: the normal vector which determines which of the two solutions is kept.
    """
    if np.isclose(x2, x1):
        r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3 = swap_beads(r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3)
    plane, r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3 = swap_axes(r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3)
    if np.isclose((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1), 0):
        raise ValueError("No solution can be computed.")
    x, y, z = naive_resting_bead(r, r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3, s=+1)
    a1 = are_aligned(np.array((x, y, z)), np.array((x1, y1, z1)), gravity)
    a2 = are_aligned(np.array((x, y, z)), np.array((x1, y1, z1)), gravity)
    a3 = are_aligned(np.array((x, y, z)), np.array((x1, y1, z1)), gravity)

    if a1 and a2  and a3:
        return restore_swapped_axes(plane, x, y, z)

    x, y, z = naive_resting_bead(r, r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3, s=-1)
    a1 = are_aligned(np.array((x, y, z)), np.array((x1, y1, z1)), gravity)
    a2 = are_aligned(np.array((x, y, z)), np.array((x1, y1, z1)), gravity)
    a3 = are_aligned(np.array((x, y, z)), np.array((x1, y1, z1)), gravity)

    if a1 and a2 and a3:
        return restore_swapped_axes(plane, x, y, z)
    
    raise ValueError(rf"¯\_(ツ)_/¯: bead cannot be properly oriented to resist {gravity = }")


def main():
    import pyvista as pv

    x1, y1, z1, r1 = 0, 0, 1, 1
    x2, y2, z2, r2 = 2, 2, -2, 2
    x3, y3, z3, r3 = 3, -3, -1, 2
    r = 3

    p = pv.Plotter()
    s1 = pv.Sphere(radius=r1, center=(x1, y1, z1))
    s2 = pv.Sphere(radius=r2, center=(x2, y2, z2))
    s3 = pv.Sphere(radius=r3, center=(x3, y3, z3))

    for s in (s1, s2, s3):
        p.add_mesh(s)

    x_s, y_s, z_s = resting_bead(r, r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3, (0, 0, -1))

    p.add_mesh(pv.Sphere(radius=r, center=(x_s, y_s, z_s)), color="red")
    p.show()


if __name__ == "__main__":
    main()