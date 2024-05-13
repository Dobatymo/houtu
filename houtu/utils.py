import math
from typing import Callable, Iterator, Tuple

import numpy as np


def rand_lat_lon(samples: int, form: str = "radians", dtype=np.float32) -> np.ndarray:
    """Sample a list of random GPS coordinates, assuming the earth is a perfect sphere."""

    arr = np.random.uniform(0, 1, (samples, 2)).astype(dtype)

    if form == "radians":
        arr[:, 0] = np.arccos(2 * arr[:, 0] - 1) - np.pi * 0.5
        arr[:, 1] *= 2 * np.pi
    elif form == "degrees":
        arr[:, 0] = 180.0 / np.pi * np.arccos(2 * arr[:, 0] - 1) - 90.0
        arr[:, 1] *= 180.0
    else:
        raise ValueError(f"Invalid input form: {form}")

    return arr


def rand_sphere_points_1(samples: int, dtype=np.float32) -> np.ndarray:
    arr = np.random.normal(size=(samples, 3)).astype(dtype)
    norms = np.sqrt(np.einsum("ij,ij->i", arr, arr, optimize=True)).reshape(-1, 1)
    return arr / norms


def rand_sphere_points_2(samples: int, dtype=np.float32) -> np.ndarray:
    p = np.random.uniform(0, 2 * np.pi, (samples,)).astype(dtype)
    u = np.random.uniform(-1, 1, (samples,)).astype(dtype)

    norm = np.sqrt(1 - u * u)
    x = norm * np.cos(p)
    y = norm * np.sin(p)
    z = u

    return np.stack([x, y, z], axis=-1)


rand_sphere_points = rand_sphere_points_2


def golden_section_search(
    func: Callable[[float], float], a: float, b: float, tol: float
) -> Iterator[Tuple[float, float]]:
    """Golden-section search.

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.

    Example:
    >>> f = lambda x: (x-2)**2
    >>> a = 1
    >>> b = 5
    >>> tol = 1e-5
    >>> (c,d) = gss(f, a, b, tol)
    >>> print(c, d)
    1.9999959837979107 2.0000050911830893
    """

    assert b > a

    h = b - a
    if h <= tol:
        yield (a, b)
        return

    invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
    invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2

    # Required steps to achieve tolerance
    n = int(math.ceil(math.log(tol / h) / math.log(invphi)))

    c = a + invphi2 * h
    d = a + invphi * h
    yc = func(c)
    yd = func(d)

    for _k in range(n - 1):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = func(c)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = func(d)

        if yc < yd:
            yield (a, d)
        else:
            yield (c, b)
