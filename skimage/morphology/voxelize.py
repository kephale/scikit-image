"""voxelize.py - voxelization

This module provides functions for voxelizing a 3D mesh.
Voxelization is the creation of an image from a mesh input.

For more information on voxelization see:
Aleksandrov, M., Zlatanova, S., & Heslop, D. J. (2021). Voxelisation algorithms and data structures: A review. Sensors, 21(24), 8241.
"""
import numpy as np

from skimage import img_as_bool


def dot_array(v1, v2):
    return sum(i * j for i, j in zip(v1, v2))


def plane_box_overlap(normal, vert, maxbox):
    vmin = [0, 0, 0]
    vmax = [0, 0, 0]
    for q in range(3):
        v = vert[q]
        if normal[q] > 0.0:
            vmin[q] = -maxbox[q] - v
            vmax[q] = maxbox[q] - v
        else:
            vmax[q] = -maxbox[q] - v
            vmin[q] = maxbox[q] - v

    if dot_array(normal, vmin) > 0.0:
        return 0
    if dot_array(normal, vmax) >= 0.0:
        return 1
    return 0


def axis_test(e, v0, v1, v2, boxhalfsize):
    p = [0, 0, 0, 0]
    p[0] = e[0] * v0[1] - e[1] * v0[2]
    p[1] = e[0] * v1[1] - e[1] * v1[2]
    p[2] = e[0] * v2[1] - e[1] * v2[2]
    rad = abs(e[0]) * boxhalfsize[1] + abs(e[1]) * boxhalfsize[2]
    if min(p) > rad or max(p) < -rad:
        return 0
    return 1


def sub(v0, vert1):
    return [v1 - v0[i] for i, v1 in enumerate(vert1)]


def cross(v1, v2):
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ]


def tri_box_overlap(boxcenter, boxhalfsize, pf1, pf2, pf3):
    v0 = sub(boxcenter, pf1)
    v1 = sub(boxcenter, pf2)
    v2 = sub(boxcenter, pf3)

    e0 = sub(v1, v0)
    e1 = sub(v2, v1)
    e2 = sub(v0, v2)

    if not axis_test(e0, v0, v1, v2, boxhalfsize):
        return 0
    if not axis_test(e1, v0, v1, v2, boxhalfsize):
        return 0
    if not axis_test(e2, v0, v1, v2, boxhalfsize):
        return 0

    min_val = min(v0[0], v1[0], v2[0])
    max_val = max(v0[0], v1[0], v2[0])
    if min_val > boxhalfsize[0] or max_val < -boxhalfsize[0]:
        return 0

    min_val = min(v0[1], v1[1], v2[1])
    max_val = max(v0[1], v1[1], v2[1])
    if min_val > boxhalfsize[1] or max_val < -boxhalfsize[1]:
        return 0

    min_val = min(v0[2], v1[2], v2[2])
    max_val = max(v0[2], v1[2], v2[2])
    if min_val > boxhalfsize[2] or max_val < -boxhalfsize[2]:
        return 0

    normal = cross(e0, e1)
    if not plane_box_overlap(normal, v0, boxhalfsize):
        return 0

    return 1


def ray_intersects_triangle(p, d, v0, v1, v2):
    """Check if ray intersects triangle."""
    e1 = v1 - v0
    e2 = v2 - v0
    h = np.cross(d, e2)
    a = np.dot(e1, h)

    if a > -1e-20 and a < 1e-20:
        return False

    f = 1.0 / a
    s = p - v0
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return False

    q = np.cross(s, e1)
    v = f * np.dot(d, q)

    if v < 0.0 or u + v > 1.0:
        return False

    t = f * np.dot(e2, q)
    if t > 1e-20:
        return True

    return False


def voxelize_mesh(vertices, faces, shape):
    """
    Convert a mesh into a binary volumetric representation.

    Parameters:
    - vertices : (V, 3) array
        Vertices of the mesh.

    - faces : (F, 3) array
        Faces (triangles) of the mesh. Each row represents a triangle with indices into vertices.

    - shape : tuple of int
        Shape of the output volume. Must be 3-dimensional.

    Returns:
    - volume : ndarray
        Binary volume of the specified shape. Voxels on the surface are set to True.
    """
    depth, height, width = shape
    out_img = np.zeros(shape, dtype=bool)

    # Add a slight padding to ensure voxels on the edge are considered
    padding = np.array([0.5, 0.5, 0.5])
    min_point = np.min(vertices, axis=0) - padding
    max_point = np.max(vertices, axis=0) + padding

    dim_point = max_point - min_point
    step_sizes = dim_point / np.array([width, height, depth])
    voxel_halfsize = step_sizes / 2.0

    for tri in faces:
        v0 = vertices[tri[0]]
        v1 = vertices[tri[1]]
        v2 = vertices[tri[2]]

        min_sub_boundary = np.min([v0, v1, v2], axis=0) - min_point
        max_sub_boundary = np.max([v0, v1, v2], axis=0) - min_point

        indices_range = [
            (
                max(0, int(np.floor(min_sub_boundary[0] / step_sizes[0]))),
                min(int(np.ceil(max_sub_boundary[0] / step_sizes[0])) + 1, width),
            ),
            (
                max(0, int(np.floor(min_sub_boundary[1] / step_sizes[1]))),
                min(int(np.ceil(max_sub_boundary[1] / step_sizes[1])) + 1, height),
            ),
            (
                max(0, int(np.floor(min_sub_boundary[2] / step_sizes[2]))),
                min(int(np.ceil(max_sub_boundary[2] / step_sizes[2])) + 1, depth),
            ),
        ]

        # For each voxel, instead of just checking the center,
        # let's check multiple points inside the voxel for a more accurate intersection
        corners = np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ]
        )

        for i in range(*indices_range[0]):
            for j in range(*indices_range[1]):
                for k in range(*indices_range[2]):
                    if not out_img[k, j, i]:
                        voxel_center = np.array([i, j, k]) * step_sizes + voxel_halfsize
                        intersected = any(
                            tri_box_overlap(
                                voxel_center + corner, voxel_halfsize, v0, v1, v2
                            )
                            for corner in corners
                        )
                        if intersected:
                            out_img[k, j, i] = True

    return np.transpose(
        img_as_bool(out_img), (2, 1, 0)
    )  # Convert the result to a boolean image


def compute_shape_from_mesh(verts, voxel_size):
    """Compute the shape of the voxel grid based on the mesh's bounding box and a given voxel size."""
    min_point = np.min(verts, axis=0)
    max_point = np.max(verts, axis=0)
    range_point = max_point - min_point
    return tuple((range_point / voxel_size).astype(int))


# if __name__ == "__main__":
#     import napari

#     # Generate a level set about zero of two identical ellipsoids in 3D
#     ellip_base = ellipsoid(6, 10, 16, levelset=True)
#     ellip_double = np.concatenate((ellip_base[:-1, ...], ellip_base[2:, ...]), axis=0)

#     # Use marching cubes to obtain the surface mesh of these ellipsoids
#     verts, faces, normals, values = measure.marching_cubes(ellip_double, 0)

#     viewer = napari.Viewer()

#     # Add the surface (verts, faces, and values) to the viewer
#     surface_layer = viewer.add_surface(
#         (verts, faces, values), colormap="blue", name="ellipsoids"
#     )

#     voxelized_img = voxelize_mesh(verts, faces, shape=compute_shape_from_mesh(verts, 1))
#     # voxelized_img = voxelize_mesh(verts, faces, shape=(270, 230, 350))

#     viewer.add_image(voxelized_img)
