import numpy as np
from scipy.spatial import Voronoi
import config

def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Input data must have 2 dimensions.")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges.get(p1, [])
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def _line_intersect_x(a, b, x):
    t = (x - a[0]) / (b[0] - a[0])
    return (x, a[1] + t * (b[1] - a[1]))


def _line_intersect_y(a, b, y):
    t = (y - a[1]) / (b[1] - a[1])
    return (a[0] + t * (b[0] - a[0]), y)


def clip_polygon(poly, xmin, xmax, ymin, ymax):
    def clip_edge(points, inside_fn, intersect_fn):
        if not points:
            return []
        result = []
        prev = points[-1]
        prev_in = inside_fn(prev)
        for curr in points:
            curr_in = inside_fn(curr)
            if curr_in:
                if not prev_in:
                    result.append(intersect_fn(prev, curr))
                result.append(curr)
            elif prev_in:
                result.append(intersect_fn(prev, curr))
            prev, prev_in = curr, curr_in
        return result

    poly = clip_edge(poly, lambda p: p[0] >= xmin, lambda a, b: _line_intersect_x(a, b, xmin))
    poly = clip_edge(poly, lambda p: p[0] <= xmax, lambda a, b: _line_intersect_x(a, b, xmax))
    poly = clip_edge(poly, lambda p: p[1] >= ymin, lambda a, b: _line_intersect_y(a, b, ymin))
    poly = clip_edge(poly, lambda p: p[1] <= ymax, lambda a, b: _line_intersect_y(a, b, ymax))
    return poly


def _create_reflected_boundaries(real_points, xmin, xmax, ymin, ymax):
    """
    Reflects every real coordinate point across the 4 boundary axes and 4 corners.
    This operation creates artificial boundary points. These boundary points prevent
    the internal cells from connecting to distant nodes. This ensures cell dimensions
    remain uniform at the grid boundaries.
    """
    x, y = real_points[:, 0], real_points[:, 1]
    reflected_points = [
        np.column_stack([2 * xmin - x, y]),
        np.column_stack([2 * xmax - x, y]),
        np.column_stack([x, 2 * ymin - y]),
        np.column_stack([x, 2 * ymax - y]),
        np.column_stack([2 * xmin - x, 2 * ymin - y]),
        np.column_stack([2 * xmin - x, 2 * ymax - y]),
        np.column_stack([2 * xmax - x, 2 * ymin - y]),
        np.column_stack([2 * xmax - x, 2 * ymax - y]),
    ]
    return np.vstack([real_points] + reflected_points)


def _polygon_centroid(poly):
    poly = np.asarray(poly)
    if len(poly) < 3:
        return poly.mean(axis=0)
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area2 = cross.sum()
    if abs(area2) < 1e-9:
        return poly.mean(axis=0)
    cx = ((x + x1) * cross).sum() / (3 * area2)
    cy = ((y + y1) * cross).sum() / (3 * area2)
    return np.array([cx, cy])


def lloyd_relax_points(points, base_idx, iterations, xmin, xmax, ymin, ymax):
    pts = points.copy()
    for _ in range(iterations):
        mirrored = _create_reflected_boundaries(pts, xmin, xmax, ymin, ymax)
        vor = Voronoi(mirrored)
        regions, vertices = voronoi_finite_polygons_2d(vor)

        new_pts = pts.copy()
        for i in range(len(pts)):
            if i == base_idx:
                continue
            poly = [tuple(vertices[v]) for v in regions[i]]
            poly = clip_polygon(poly, xmin, xmax, ymin, ymax)
            if len(poly) >= 3:
                new_pts[i] = _polygon_centroid(poly)
        pts = new_pts
    return pts


def build_adjacency(vor, n_points):
    """
    The parameter n_points defines the number of valid nodes. The function
    ignores any ridge that connects to a reflected boundary point (index >= n_points).
    """
    adjacency = {i: set() for i in range(n_points)}
    for p1, p2 in vor.ridge_points:
        if p1 < n_points and p2 < n_points:
            adjacency[p1].add(int(p2))
            adjacency[p2].add(int(p1))
    return adjacency


def generate_points(rng, n_seeds):
    base_point = np.array([[0.6, 0.6]])
    margin = 0.4
    others = rng.uniform(config.BOUNDS[0] + margin, config.BOUNDS[1] - margin, size=(n_seeds - 1, 2))
    return np.vstack([base_point, others])