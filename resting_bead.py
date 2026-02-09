import numpy as np


def resting_bead_zplus(r, r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3):
    z = (2*z1*(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 + (((-2*z1*(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 + (x1*z2 - x1*z3 - x2*z1 + x2*z3 + x3*z1 - x3*z2)*(-2*r*r1*x2 + 2*r*r1*x3 + 2*r*r2*x1 - 2*r*r2*x3 - 2*r*r3*x1 + 2*r*r3*x2 - r1**2*x2 + r1**2*x3 + r2**2*x1 - r2**2*x3 - r3**2*x1 + r3**2*x2 + x1**2*x2 - x1**2*x3 - x1*x2**2 + x1*x3**2 + 2*x1*y1*y2 - 2*x1*y1*y3 - x1*y2**2 + x1*y3**2 - x1*z2**2 + x1*z3**2 + x2**2*x3 - x2*x3**2 - x2*y1**2 + 2*x2*y1*y3 - x2*y3**2 + x2*z1**2 - x2*z3**2 + x3*y1**2 - 2*x3*y1*y2 + x3*y2**2 - x3*z1**2 + x3*z2**2) + (-y1*z2 + y1*z3 + y2*z1 - y2*z3 - y3*z1 + y3*z2)*(2*r*r1*y2 - 2*r*r1*y3 - 2*r*r2*y1 + 2*r*r2*y3 + 2*r*r3*y1 - 2*r*r3*y2 + r1**2*y2 - r1**2*y3 - r2**2*y1 + r2**2*y3 + r3**2*y1 - r3**2*y2 + x1**2*y2 - x1**2*y3 - 2*x1*x2*y1 + 2*x1*x2*y3 + 2*x1*x3*y1 - 2*x1*x3*y2 + x2**2*y1 - x2**2*y3 - x3**2*y1 + x3**2*y2 - y1**2*y2 + y1**2*y3 + y1*y2**2 - y1*y3**2 + y1*z2**2 - y1*z3**2 - y2**2*y3 + y2*y3**2 - y2*z1**2 + y2*z3**2 + y3*z1**2 - y3*z2**2))**2 - ((-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 + (x1*z2 - x1*z3 - x2*z1 + x2*z3 + x3*z1 - x3*z2)**2 + (-y1*z2 + y1*z3 + y2*z1 - y2*z3 - y3*z1 + y3*z2)**2)*(4*z1**2*(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 - 4*(r + r1)**2*(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 + (-2*r*r1*x2 + 2*r*r1*x3 + 2*r*r2*x1 - 2*r*r2*x3 - 2*r*r3*x1 + 2*r*r3*x2 - r1**2*x2 + r1**2*x3 + r2**2*x1 - r2**2*x3 - r3**2*x1 + r3**2*x2 + x1**2*x2 - x1**2*x3 - x1*x2**2 + x1*x3**2 + 2*x1*y1*y2 - 2*x1*y1*y3 - x1*y2**2 + x1*y3**2 - x1*z2**2 + x1*z3**2 + x2**2*x3 - x2*x3**2 - x2*y1**2 + 2*x2*y1*y3 - x2*y3**2 + x2*z1**2 - x2*z3**2 + x3*y1**2 - 2*x3*y1*y2 + x3*y2**2 - x3*z1**2 + x3*z2**2)**2 + (2*r*r1*y2 - 2*r*r1*y3 - 2*r*r2*y1 + 2*r*r2*y3 + 2*r*r3*y1 - 2*r*r3*y2 + r1**2*y2 - r1**2*y3 - r2**2*y1 + r2**2*y3 + r3**2*y1 - r3**2*y2 + x1**2*y2 - x1**2*y3 - 2*x1*x2*y1 + 2*x1*x2*y3 + 2*x1*x3*y1 - 2*x1*x3*y2 + x2**2*y1 - x2**2*y3 - x3**2*y1 + x3**2*y2 - y1**2*y2 + y1**2*y3 + y1*y2**2 - y1*y3**2 + y1*z2**2 - y1*z3**2 - y2**2*y3 + y2*y3**2 - y2*z1**2 + y2*z3**2 + y3*z1**2 - y3*z2**2)**2))/(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**4)**0.5*(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 - (x1*z2 - x1*z3 - x2*z1 + x2*z3 + x3*z1 - x3*z2)*(-2*r*r1*x2 + 2*r*r1*x3 + 2*r*r2*x1 - 2*r*r2*x3 - 2*r*r3*x1 + 2*r*r3*x2 - r1**2*x2 + r1**2*x3 + r2**2*x1 - r2**2*x3 - r3**2*x1 + r3**2*x2 + x1**2*x2 - x1**2*x3 - x1*x2**2 + x1*x3**2 + 2*x1*y1*y2 - 2*x1*y1*y3 - x1*y2**2 + x1*y3**2 - x1*z2**2 + x1*z3**2 + x2**2*x3 - x2*x3**2 - x2*y1**2 + 2*x2*y1*y3 - x2*y3**2 + x2*z1**2 - x2*z3**2 + x3*y1**2 - 2*x3*y1*y2 + x3*y2**2 - x3*z1**2 + x3*z2**2) - (-y1*z2 + y1*z3 + y2*z1 - y2*z3 - y3*z1 + y3*z2)*(2*r*r1*y2 - 2*r*r1*y3 - 2*r*r2*y1 + 2*r*r2*y3 + 2*r*r3*y1 - 2*r*r3*y2 + r1**2*y2 - r1**2*y3 - r2**2*y1 + r2**2*y3 + r3**2*y1 - r3**2*y2 + x1**2*y2 - x1**2*y3 - 2*x1*x2*y1 + 2*x1*x2*y3 + 2*x1*x3*y1 - 2*x1*x3*y2 + x2**2*y1 - x2**2*y3 - x3**2*y1 + x3**2*y2 - y1**2*y2 + y1**2*y3 + y1*y2**2 - y1*y3**2 + y1*z2**2 - y1*z3**2 - y2**2*y3 + y2*y3**2 - y2*z1**2 + y2*z3**2 + y3*z1**2 - y3*z2**2))/(2*((-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 + (x1*z2 - x1*z3 - x2*z1 + x2*z3 + x3*z1 - x3*z2)**2 + (-y1*z2 + y1*z3 + y2*z1 - y2*z3 - y3*z1 + y3*z2)**2))
    y = ((-x1 + x2)*(-x1**2 + x3**2 - y1**2 + y3**2 + 2*z*(z1 - z3) - z1**2 + z3**2 + (r + r1)**2 - (r + r3)**2) - (-x1 + x3)*(-x1**2 + x2**2 - y1**2 + y2**2 + 2*z*(z1 - z2) - z1**2 + z2**2 + (r + r1)**2 - (r + r2)**2))/(2*((-x1 + x2)*(-y1 + y3) - (-x1 + x3)*(-y1 + y2)))
    x = (-(-y1 + y2)*((-x1 + x2)*(-x1**2 + x3**2 - y1**2 + y3**2 + 2*z*(z1 - z3) - z1**2 + z3**2 + (r + r1)**2 - (r + r3)**2) - (-x1 + x3)*(-x1**2 + x2**2 - y1**2 + y2**2 + 2*z*(z1 - z2) - z1**2 + z2**2 + (r + r1)**2 - (r + r2)**2)) + ((-x1 + x2)*(-y1 + y3) - (-x1 + x3)*(-y1 + y2))*(-x1**2 + x2**2 - y1**2 + y2**2 + 2*z*(z1 - z2) - z1**2 + z2**2 + (r + r1)**2 - (r + r2)**2))/(2*(-x1 + x2)*((-x1 + x2)*(-y1 + y3) - (-x1 + x3)*(-y1 + y2)))
    return x, y, z


def resting_bead_zminus(r, r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3):
    z = (2*z1*(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 - (((-2*z1*(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 + (x1*z2 - x1*z3 - x2*z1 + x2*z3 + x3*z1 - x3*z2)*(-2*r*r1*x2 + 2*r*r1*x3 + 2*r*r2*x1 - 2*r*r2*x3 - 2*r*r3*x1 + 2*r*r3*x2 - r1**2*x2 + r1**2*x3 + r2**2*x1 - r2**2*x3 - r3**2*x1 + r3**2*x2 + x1**2*x2 - x1**2*x3 - x1*x2**2 + x1*x3**2 + 2*x1*y1*y2 - 2*x1*y1*y3 - x1*y2**2 + x1*y3**2 - x1*z2**2 + x1*z3**2 + x2**2*x3 - x2*x3**2 - x2*y1**2 + 2*x2*y1*y3 - x2*y3**2 + x2*z1**2 - x2*z3**2 + x3*y1**2 - 2*x3*y1*y2 + x3*y2**2 - x3*z1**2 + x3*z2**2) + (-y1*z2 + y1*z3 + y2*z1 - y2*z3 - y3*z1 + y3*z2)*(2*r*r1*y2 - 2*r*r1*y3 - 2*r*r2*y1 + 2*r*r2*y3 + 2*r*r3*y1 - 2*r*r3*y2 + r1**2*y2 - r1**2*y3 - r2**2*y1 + r2**2*y3 + r3**2*y1 - r3**2*y2 + x1**2*y2 - x1**2*y3 - 2*x1*x2*y1 + 2*x1*x2*y3 + 2*x1*x3*y1 - 2*x1*x3*y2 + x2**2*y1 - x2**2*y3 - x3**2*y1 + x3**2*y2 - y1**2*y2 + y1**2*y3 + y1*y2**2 - y1*y3**2 + y1*z2**2 - y1*z3**2 - y2**2*y3 + y2*y3**2 - y2*z1**2 + y2*z3**2 + y3*z1**2 - y3*z2**2))**2 - ((-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 + (x1*z2 - x1*z3 - x2*z1 + x2*z3 + x3*z1 - x3*z2)**2 + (-y1*z2 + y1*z3 + y2*z1 - y2*z3 - y3*z1 + y3*z2)**2)*(4*z1**2*(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 - 4*(r + r1)**2*(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 + (-2*r*r1*x2 + 2*r*r1*x3 + 2*r*r2*x1 - 2*r*r2*x3 - 2*r*r3*x1 + 2*r*r3*x2 - r1**2*x2 + r1**2*x3 + r2**2*x1 - r2**2*x3 - r3**2*x1 + r3**2*x2 + x1**2*x2 - x1**2*x3 - x1*x2**2 + x1*x3**2 + 2*x1*y1*y2 - 2*x1*y1*y3 - x1*y2**2 + x1*y3**2 - x1*z2**2 + x1*z3**2 + x2**2*x3 - x2*x3**2 - x2*y1**2 + 2*x2*y1*y3 - x2*y3**2 + x2*z1**2 - x2*z3**2 + x3*y1**2 - 2*x3*y1*y2 + x3*y2**2 - x3*z1**2 + x3*z2**2)**2 + (2*r*r1*y2 - 2*r*r1*y3 - 2*r*r2*y1 + 2*r*r2*y3 + 2*r*r3*y1 - 2*r*r3*y2 + r1**2*y2 - r1**2*y3 - r2**2*y1 + r2**2*y3 + r3**2*y1 - r3**2*y2 + x1**2*y2 - x1**2*y3 - 2*x1*x2*y1 + 2*x1*x2*y3 + 2*x1*x3*y1 - 2*x1*x3*y2 + x2**2*y1 - x2**2*y3 - x3**2*y1 + x3**2*y2 - y1**2*y2 + y1**2*y3 + y1*y2**2 - y1*y3**2 + y1*z2**2 - y1*z3**2 - y2**2*y3 + y2*y3**2 - y2*z1**2 + y2*z3**2 + y3*z1**2 - y3*z2**2)**2))/(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**4)**0.5*(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 - (x1*z2 - x1*z3 - x2*z1 + x2*z3 + x3*z1 - x3*z2)*(-2*r*r1*x2 + 2*r*r1*x3 + 2*r*r2*x1 - 2*r*r2*x3 - 2*r*r3*x1 + 2*r*r3*x2 - r1**2*x2 + r1**2*x3 + r2**2*x1 - r2**2*x3 - r3**2*x1 + r3**2*x2 + x1**2*x2 - x1**2*x3 - x1*x2**2 + x1*x3**2 + 2*x1*y1*y2 - 2*x1*y1*y3 - x1*y2**2 + x1*y3**2 - x1*z2**2 + x1*z3**2 + x2**2*x3 - x2*x3**2 - x2*y1**2 + 2*x2*y1*y3 - x2*y3**2 + x2*z1**2 - x2*z3**2 + x3*y1**2 - 2*x3*y1*y2 + x3*y2**2 - x3*z1**2 + x3*z2**2) - (-y1*z2 + y1*z3 + y2*z1 - y2*z3 - y3*z1 + y3*z2)*(2*r*r1*y2 - 2*r*r1*y3 - 2*r*r2*y1 + 2*r*r2*y3 + 2*r*r3*y1 - 2*r*r3*y2 + r1**2*y2 - r1**2*y3 - r2**2*y1 + r2**2*y3 + r3**2*y1 - r3**2*y2 + x1**2*y2 - x1**2*y3 - 2*x1*x2*y1 + 2*x1*x2*y3 + 2*x1*x3*y1 - 2*x1*x3*y2 + x2**2*y1 - x2**2*y3 - x3**2*y1 + x3**2*y2 - y1**2*y2 + y1**2*y3 + y1*y2**2 - y1*y3**2 + y1*z2**2 - y1*z3**2 - y2**2*y3 + y2*y3**2 - y2*z1**2 + y2*z3**2 + y3*z1**2 - y3*z2**2))/(2*((-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)**2 + (x1*z2 - x1*z3 - x2*z1 + x2*z3 + x3*z1 - x3*z2)**2 + (-y1*z2 + y1*z3 + y2*z1 - y2*z3 - y3*z1 + y3*z2)**2))
    y = ((-x1 + x2)*(-x1**2 + x3**2 - y1**2 + y3**2 + 2*z*(z1 - z3) - z1**2 + z3**2 + (r + r1)**2 - (r + r3)**2) - (-x1 + x3)*(-x1**2 + x2**2 - y1**2 + y2**2 + 2*z*(z1 - z2) - z1**2 + z2**2 + (r + r1)**2 - (r + r2)**2))/(2*((-x1 + x2)*(-y1 + y3) - (-x1 + x3)*(-y1 + y2)))
    x = (-(-y1 + y2)*((-x1 + x2)*(-x1**2 + x3**2 - y1**2 + y3**2 + 2*z*(z1 - z3) - z1**2 + z3**2 + (r + r1)**2 - (r + r3)**2) - (-x1 + x3)*(-x1**2 + x2**2 - y1**2 + y2**2 + 2*z*(z1 - z2) - z1**2 + z2**2 + (r + r1)**2 - (r + r2)**2)) + ((-x1 + x2)*(-y1 + y3) - (-x1 + x3)*(-y1 + y2))*(-x1**2 + x2**2 - y1**2 + y2**2 + 2*z*(z1 - z2) - z1**2 + z2**2 + (r + r1)**2 - (r + r2)**2))/(2*(-x1 + x2)*((-x1 + x2)*(-y1 + y3) - (-x1 + x3)*(-y1 + y2)))
    return x, y, z


def are_aligned(p1, p2, v):
    return np.dot((p2-p1), v) > 0


def exchange_problematic_beads(r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3):
    return r3, x3, y3, z3, r2, x2, y2, z2, r1, x1, y1, z1


def resting_bead(r, r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3, gravity=np.array((0, 0, -1))):
    if np.isclose(x2, x1):
        print("Exchanging beads !")
        r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3 = exchange_problematic_beads(r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3)
    if np.isclose((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1), 0):
        raise ValueError("No solution can be computed.")
    x, y, z = resting_bead_zplus(r, r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3)
    a1 = are_aligned(np.array((x, y, z)), np.array((x1, y1, z1)), gravity)
    a2 = are_aligned(np.array((x, y, z)), np.array((x1, y1, z1)), gravity)
    a3 = are_aligned(np.array((x, y, z)), np.array((x1, y1, z1)), gravity)

    if a1 and a2  and a3:
        return x, y, z

    x, y, z = resting_bead_zminus(r, r1, x1, y1, z1, r2, x2, y2, z2, r3, x3, y3, z3)
    a1 = are_aligned(np.array((x, y, z)), np.array((x1, y1, z1)), gravity)
    a2 = are_aligned(np.array((x, y, z)), np.array((x1, y1, z1)), gravity)
    a3 = are_aligned(np.array((x, y, z)), np.array((x1, y1, z1)), gravity)

    if a1 and a2 and a3:
        return x, y, z
    
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