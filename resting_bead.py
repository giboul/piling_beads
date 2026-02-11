"""
Module for finding the position of a single sphere of known radius 
in tangent contact with three other spheres.

Use the `resting_bead` function or `naive_resting_bead` for naive calculation.
"""
import numpy as np


def naive_resting_bead(r, r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3, s=+1):
    """
    From the radius of the new sphere and the coordinates and radii of three other spheres,
    find the position of the tangent fourth sphere. This solution is not valid for cases where 
    x1 == x2 or the centers are on a vertical plane (xz or yz).

    :param r: Radius of the new sphere
    :param r1: Radius of the first sphere
    :param x1: `x` coordinate of the first sphere
    :param y1: `y` coordinate of the first sphere
    :param z1: `z` coordinate of the first sphere
    :param r2: Radius of the second sphere
    :param x2: `x` coordinate of the second sphere
    :param y2: `y` coordinate of the second sphere
    :param z2: `z` coordinate of the second sphere
    :param r3: Radius of the third sphere
    :param x3: `x` coordinate of the third sphere
    :param y3: `y` coordinate of the third sphere
    :param z3: `z` coordinate of the third sphere
    :param s: Which solution to keep: s=+1 top solution, s=-1 -> bottom solution
    """
    z = (s*(((-2*z1*(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 + (x1*z2 - x1*z3 - x2*z1 + x2*z3 + x3*z1 - x3*z2)*(-2*r*r1*x2 + 2*r*r1*x3 + 2*r*r2*x1 - 2*r*r2*x3 - 2*r*r3*x1 + 2*r*r3*x2 - r1**2*x2 + r1**2*x3 + r2**2*x1 - r2**2*x3 - r3**2*x1 + r3**2*x2 + x1**2*x2 - x1**2*x3 - x1*x2**2 + x1*x3**2 + 2*x1*y1*y2 - 2*x1*y1*y3 - x1*y2**2 + x1*y3**2 - x1*z2**2 + x1*z3**2 + x2**2*x3 - x2*x3**2 - x2*y1**2 + 2*x2*y1*y3 - x2*y3**2 + x2*z1**2 - x2*z3**2 + x3*y1**2 - 2*x3*y1*y2 + x3*y2**2 - x3*z1**2 + x3*z2**2) + (-y1*z2 + y1*z3 + y2*z1 - y2*z3 - y3*z1 + y3*z2)*(2*r*r1*y2 - 2*r*r1*y3 - 2*r*r2*y1 + 2*r*r2*y3 + 2*r*r3*y1 - 2*r*r3*y2 + r1**2*y2 - r1**2*y3 - r2**2*y1 + r2**2*y3 + r3**2*y1 - r3**2*y2 + x1**2*y2 - x1**2*y3 - 2*x1*x2*y1 + 2*x1*x2*y3 + 2*x1*x3*y1 - 2*x1*x3*y2 + x2**2*y1 - x2**2*y3 - x3**2*y1 + x3**2*y2 - y1**2*y2 + y1**2*y3 + y1*y2**2 - y1*y3**2 + y1*z2**2 - y1*z3**2 - y2**2*y3 + y2*y3**2 - y2*z1**2 + y2*z3**2 + y3*z1**2 - y3*z2**2))**2 - ((-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 + (x1*z2 - x1*z3 - x2*z1 + x2*z3 + x3*z1 - x3*z2)**2 + (-y1*z2 + y1*z3 + y2*z1 - y2*z3 - y3*z1 + y3*z2)**2)*(4*z1**2*(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 - 4*(r + r1)**2*(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 + (-2*r*r1*x2 + 2*r*r1*x3 + 2*r*r2*x1 - 2*r*r2*x3 - 2*r*r3*x1 + 2*r*r3*x2 - r1**2*x2 + r1**2*x3 + r2**2*x1 - r2**2*x3 - r3**2*x1 + r3**2*x2 + x1**2*x2 - x1**2*x3 - x1*x2**2 + x1*x3**2 + 2*x1*y1*y2 - 2*x1*y1*y3 - x1*y2**2 + x1*y3**2 - x1*z2**2 + x1*z3**2 + x2**2*x3 - x2*x3**2 - x2*y1**2 + 2*x2*y1*y3 - x2*y3**2 + x2*z1**2 - x2*z3**2 + x3*y1**2 - 2*x3*y1*y2 + x3*y2**2 - x3*z1**2 + x3*z2**2)**2 + (2*r*r1*y2 - 2*r*r1*y3 - 2*r*r2*y1 + 2*r*r2*y3 + 2*r*r3*y1 - 2*r*r3*y2 + r1**2*y2 - r1**2*y3 - r2**2*y1 + r2**2*y3 + r3**2*y1 - r3**2*y2 + x1**2*y2 - x1**2*y3 - 2*x1*x2*y1 + 2*x1*x2*y3 + 2*x1*x3*y1 - 2*x1*x3*y2 + x2**2*y1 - x2**2*y3 - x3**2*y1 + x3**2*y2 - y1**2*y2 + y1**2*y3 + y1*y2**2 - y1*y3**2 + y1*z2**2 - y1*z3**2 - y2**2*y3 + y2*y3**2 - y2*z1**2 + y2*z3**2 + y3*z1**2 - y3*z2**2)**2))/(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**4)**0.5*(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 + 2*z1*(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 - (x1*z2 - x1*z3 - x2*z1 + x2*z3 + x3*z1 - x3*z2)*(-2*r*r1*x2 + 2*r*r1*x3 + 2*r*r2*x1 - 2*r*r2*x3 - 2*r*r3*x1 + 2*r*r3*x2 - r1**2*x2 + r1**2*x3 + r2**2*x1 - r2**2*x3 - r3**2*x1 + r3**2*x2 + x1**2*x2 - x1**2*x3 - x1*x2**2 + x1*x3**2 + 2*x1*y1*y2 - 2*x1*y1*y3 - x1*y2**2 + x1*y3**2 - x1*z2**2 + x1*z3**2 + x2**2*x3 - x2*x3**2 - x2*y1**2 + 2*x2*y1*y3 - x2*y3**2 + x2*z1**2 - x2*z3**2 + x3*y1**2 - 2*x3*y1*y2 + x3*y2**2 - x3*z1**2 + x3*z2**2) - (-y1*z2 + y1*z3 + y2*z1 - y2*z3 - y3*z1 + y3*z2)*(2*r*r1*y2 - 2*r*r1*y3 - 2*r*r2*y1 + 2*r*r2*y3 + 2*r*r3*y1 - 2*r*r3*y2 + r1**2*y2 - r1**2*y3 - r2**2*y1 + r2**2*y3 + r3**2*y1 - r3**2*y2 + x1**2*y2 - x1**2*y3 - 2*x1*x2*y1 + 2*x1*x2*y3 + 2*x1*x3*y1 - 2*x1*x3*y2 + x2**2*y1 - x2**2*y3 - x3**2*y1 + x3**2*y2 - y1**2*y2 + y1**2*y3 + y1*y2**2 - y1*y3**2 + y1*z2**2 - y1*z3**2 - y2**2*y3 + y2*y3**2 - y2*z1**2 + y2*z3**2 + y3*z1**2 - y3*z2**2))/(2*((-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 + (x1*z2 - x1*z3 - x2*z1 + x2*z3 + x3*z1 - x3*z2)**2 + (-y1*z2 + y1*z3 + y2*z1 - y2*z3 - y3*z1 + y3*z2)**2))
    # y = ((-x1 + x2)*(-x1**2 + x3**2 - y1**2 + y3**2 + 2*z*(z1 - z3) - z1**2 + z3**2 + (r + r1)**2 - (r + r3)**2) - (-x1 + x3)*(-x1**2 + x2**2 - y1**2 + y2**2 + 2*z*(z1 - z2) - z1**2 + z2**2 + (r + r1)**2 - (r + r2)**2))/(2*((-x1 + x2)*(-y1 + y3) - (-x1 + x3)*(-y1 + y2)))
    # x = (-(-y1 + y2)*((-x1 + x2)*(-x1**2 + x3**2 - y1**2 + y3**2 + 2*z*(z1 - z3) - z1**2 + z3**2 + (r + r1)**2 - (r + r3)**2) - (-x1 + x3)*(-x1**2 + x2**2 - y1**2 + y2**2 + 2*z*(z1 - z2) - z1**2 + z2**2 + (r + r1)**2 - (r + r2)**2)) + ((-x1 + x2)*(-y1 + y3) - (-x1 + x3)*(-y1 + y2))*(-x1**2 + x2**2 - y1**2 + y2**2 + 2*z*(z1 - z2) - z1**2 + z2**2 + (r + r1)**2 - (r + r2)**2))/(2*(-x1 + x2)*((-x1 + x2)*(-y1 + y3) - (-x1 + x3)*(-y1 + y2)))
    y = ((-2*x1 + 2*x2)*(-x1**2 + x3**2 - y1**2 + y3**2 + z*(2*z1 - 2*z3) - z1**2 + z3**2 + (r + r1)**2 - (r + r3)**2) - (-2*x1 + 2*x3)*(-x1**2 + x2**2 - y1**2 + y2**2 + z*(2*z1 - 2*z2) - z1**2 + z2**2 + (r + r1)**2 - (r + r2)**2))/((-2*x1 + 2*x2)*(-2*y1 + 2*y3) - (-2*x1 + 2*x3)*(-2*y1 + 2*y2))
    x = -y*(-2*y1 + 2*y2)/(-2*x1 + 2*x2) + (-x1**2 + x2**2 - y1**2 + y2**2 + z*(2*z1 - 2*z2) - z1**2 + z2**2 + (r + r1)**2 - (r + r2)**2)/(-2*x1 + 2*x2)
    return x, y, z


def swap_axes(r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3, g):
    """Work in a plane where the linear solution exists."""
    _g = np.array(g, copy=True)
    if not np.isclose((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1), 0):
        return "xy", r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3, _g
    if not np.isclose((z2-z1)*(y3-y1) - (z3-z1)*(y2-y1), 0):
        _g[0], _g[2] = _g[2], _g[0]
        return "zy", r1, z1, y1, x1, r2, z2, y2, x2, r3, z3, y3, x3, _g
    if not np.isclose((x2-x1)*(z3-z1) - (x3-x1)*(z2-z1), 0):
        _g[1], _g[2] = _g[2], _g[1]
        return "xz", r1, x1, z1, y1, r2, x2, z2, y3, r3, x3, z3, y3, _g
    raise ValueError(r"¯\_(ツ)_/¯")


def restore_swapped_axes(plane, x, y, z):
    """Restore the original axes from the `swap_axes` transform."""
    if plane == "xy":
        return x, y, z
    if plane == "xz":
        return x, z, y
    if plane == "zy":
        return z, y, x
    raise ValueError(r"¯\_(ツ)_/¯")


def is_positive_linear_combination(v1, v2, v3, w):
    """
    Check that `w` can be written as a1*v1 + a2*v2 + a3*v3 with ai >= 0.

    :param v1: vector
    :param v2: vector
    :param v3: vector
    :param w: vector
    :return: True or False
    :rtype: bool
    """
    M = np.array([v1, v2, v3]).T
    x = np.linalg.solve(M, w)
    return ((x>=0) | np.isclose(x, 0)).all()


def resting_bead(r, r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3, gravity=np.array((0, 0, -1))):
    """
    From the radius of the new sphere and the coordinates and radii of three other spheres,
    find the position of the tangent fourth sphere.

    :param r: Radius of the new sphere
    :param r1: Radius of the first sphere
    :param x1: `x` coordinate of the first sphere
    :param y1: `y` coordinate of the first sphere
    :param z1: `z` coordinate of the first sphere
    :param r2: Radius of the second sphere
    :param x2: `x` coordinate of the second sphere
    :param y2: `y` coordinate of the second sphere
    :param z2: `z` coordinate of the second sphere
    :param r3: Radius of the third sphere
    :param x3: `x` coordinate of the third sphere
    :param y3: `y` coordinate of the third sphere
    :param z3: `z` coordinate of the third sphere
    :param gravity: the normal vector which determines which of the two solutions is kept.
    """
    if np.isclose(x2, x1):  # Swap first and last bead
        r1, x1, y1, z1, r3, x3, y3, z3 = r3, x3, y3, z3, r1, x1, y1, z1

    plane,r1,x1,y1,z1,r2,x2,y2,z2,r3,x3,y3,z3,gravity=swap_axes(r1,x1,y1,z1,r2,x2,y2,z2,r3,x3,y3,z3,gravity)

    xyz = np.array(naive_resting_bead(r, r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3, s=+1))
    if is_positive_linear_combination((x1, y1, z1)-xyz,
                                      (x2, y2, z2)-xyz,
                                      (x3, y3, z3)-xyz,
                                      gravity):
        return restore_swapped_axes(plane, *xyz)

    xyz = np.array(naive_resting_bead(r, r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3, s=-1))
    if is_positive_linear_combination((x1, y1, z1)-xyz,
                                      (x2, y2, z2)-xyz,
                                      (x3, y3, z3)-xyz,
                                      gravity):
        return restore_swapped_axes(plane, *xyz)

    raise ValueError(rf"¯\_(ツ)_/¯: bead cannot be properly oriented to resist {gravity=}")


def main():
    import pyvista as pv

    x1, y1, z1, r1 = 0, 0, 0, 2
    x2, y2, z2, r2 = 2, 0, 0, 2
    x3, y3, z3, r3 = 0, 2, 0, 2
    r = 2

    p = pv.Plotter()
    s1 = pv.Sphere(radius=r1, center=(x1, y1, z1))
    s2 = pv.Sphere(radius=r2, center=(x2, y2, z2))
    s3 = pv.Sphere(radius=r3, center=(x3, y3, z3))

    for s in (s1, s2, s3):
        p.add_mesh(s)

    x_s, y_s, z_s = resting_bead(
        r, r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3, (0, 0, -1))

    p.add_mesh(pv.Sphere(radius=r, center=(x_s, y_s, z_s)), color="red")
    p.show()


if __name__ == "__main__":
    main()
