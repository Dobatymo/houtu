from typing import Dict

import numpy as np
from geopy.distance import geodesic
from pymap3d.vincenty import vdist

from houtu.geocoding import ReverseGeocodeBallHaversine, ReverseGeocodeBase, ReverseGeocodeKdScipy
from houtu.utils import golden_section_search, rand_lat_lon


def geodesic_distances(query_arr: np.ndarray, coords: np.ndarray) -> np.ndarray:
    assert coords.ndim == 3
    assert coords.shape[-1] == 2
    assert coords.dtype in (np.float32, np.float64)
    assert query_arr.shape == (coords.shape[0], 2)

    out = np.empty(coords.shape[:2], dtype=np.float64)

    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            out[i, j] = geodesic(query_arr[i], coords[i, j]).meters
    return out


def vincenty_distances(query_arr: np.ndarray, coords: np.ndarray) -> np.ndarray:
    vincenty_dist, _ = vdist(query_arr[:, None, 0], query_arr[:, None, 1], coords[:, :, 0], coords[:, :, 1])
    return vincenty_dist


def measure_error(rg: ReverseGeocodeBase, query_arr: np.ndarray, true_dist: np.ndarray) -> Dict[str, float]:
    coords, distances, cities = rg.query(query_arr, k=2, form="degrees", return_distance=True)
    diff = np.abs(distances - true_dist)

    return {
        "mean_absolute_error": np.mean(diff).item(),
        "mean_relative_error": np.mean(diff / true_dist).item(),
    }


if __name__ == "__main__":
    rg_haversine = ReverseGeocodeBallHaversine()
    for _ in range(3):
        query_arr = rand_lat_lon(50000, "degrees")
        coords, cities = rg_haversine.query(query_arr, k=2, form="degrees", return_distance=False)
        coords = np.rad2deg(coords)
        vincenty_dist = vincenty_distances(query_arr, coords)
        geodesic_dist = geodesic_distances(query_arr, coords)

        true_dists = [("vincenty", vincenty_dist), ("geodesic", geodesic_dist)]

        # search
        for dist_name, true_dist in true_dists:
            for error_name in ["mean_relative_error", "mean_absolute_error"]:
                print(dist_name, error_name)

                def func(radius: float) -> float:
                    rg_haversine.radius = radius  # noqa: B023
                    errors = measure_error(rg_haversine, query_arr, true_dist)  # noqa: B023
                    return errors[error_name]  # noqa: B023

                a, b = 6000000, 7000000  # radius in meters
                print(a, b, end="")
                for a, b in golden_section_search(func, a, b, 0.1):  # search to a precision of 1mm  # noqa: B020
                    print(f"\r{a} {b}", end="")
                print()

        for name, true_dist in true_dists:
            rg_euclidic = ReverseGeocodeKdScipy()
            errors = measure_error(rg_euclidic, query_arr, true_dist)
            print(name, "euclidic", errors)
