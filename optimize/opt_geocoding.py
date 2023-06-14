from operator import itemgetter
from typing import Dict

import numpy as np
from geopy.distance import geodesic
from pymap3d.vincenty import vdist

from houtu.geocoding import ReverseGeocodeBallHaversine, ReverseGeocodeKdScipy, earth_radii
from houtu.utils import rand_lat_lon


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


def measure_error_haversine(rg, radius: float, query_arr: np.ndarray, true_dist: np.ndarray) -> Dict[str, float]:
    rg.radius = radius
    coords, distances, cities = rg.query(query_arr, k=2, form="degrees")
    diff = np.abs(distances - true_dist)

    return {
        "mean_absolute_error": np.mean(diff).item(),
        "mean_relative_error": np.mean(diff / true_dist).item(),
    }


def measure_error_euclidic(rg, query_arr: np.ndarray, true_dist: np.ndarray) -> Dict[str, float]:
    coords, distances, cities = rg.query(query_arr, k=2, form="degrees")
    diff = np.abs(distances - true_dist)

    return {
        "mean_absolute_error": np.mean(diff).item(),
        "mean_relative_error": np.mean(diff / true_dist).item(),
    }


if __name__ == "__main__":
    rg = ReverseGeocodeBallHaversine()
    query_arr = rand_lat_lon(50000, "degrees")
    coords, cities = rg.query(query_arr, k=2, form="degrees", return_distance=False)
    coords = np.rad2deg(coords)
    vincenty_dist = vincenty_distances(query_arr, coords)
    geodesic_dist = geodesic_distances(query_arr, coords)

    true_dists = [("vincenty", vincenty_dist), ("geodesic", geodesic_dist)]

    for dist_name, true_dist in true_dists:
        print(dist_name, "haversine")
        results = {}
        for name, radius in earth_radii.items():
            errors = measure_error_haversine(rg, radius, query_arr, vincenty_dist)
            for metric, error in errors.items():
                results.setdefault(metric, {})[name] = error

        for metric, errors in results.items():
            name, error = min(errors.items(), key=itemgetter(1))
            print(metric, name, error)

    for name, true_dist in true_dists:
        rg = ReverseGeocodeKdScipy()
        errors = measure_error_euclidic(rg, query_arr, true_dist)
        print(name, "euclidic", errors)
