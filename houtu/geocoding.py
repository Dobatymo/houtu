import csv
import lzma
from typing import List, Literal, NamedTuple, Optional, Set, Tuple, overload

import numpy as np
from importlib_resources import files
from sklearn.metrics.pairwise import euclidean_distances, haversine_distances


class City(NamedTuple):
    name: str
    feature_class: str
    feature_code: str
    country_code: str
    admin1_code: str
    admin2_code: str
    admin3_code: str
    admin4_code: str
    population: int
    elevation: Optional[int]
    timezone: str


class Cities(NamedTuple):
    coords: np.ndarray
    info: List[City]


earth_radii = {
    'IAU nominal "zero tide" equatorial': 6378100,
    'IAU nominal "zero tide" polar': 6356800,
    "IUGG equatorial radius": 6378137,
    "IUGG semiminor axis (b)": 6356752.3141,
    "IUGG polar radius of curvature (c)": 6399593.6259,
    "IUGG mean radius (R1)": 6371008.7714,
    "IUGG radius of sphere of same surface (R2)": 6371007.1810,
    "IUGG radius of sphere of same volume (R3)": 6371000.7900,
    "IERS WGS-84 ellipsoid, semi-major axis (a)": 6378137.0,
    "IERS WGS-84 ellipsoid, semi-minor axis (b)": 6356752.3142,
    "IERS WGS-84 ellipsoid, polar radius of curvature (c)": 6399593.6258,
    "IERS WGS-84 ellipsoid, Mean radius of semi-axes (R1)": 6371008.7714,
    "IERS WGS-84 ellipsoid, Radius of Sphere of Equal Area (R2)": 6371007.1809,
    "IERS WGS-84 ellipsoid, Radius of Sphere of Equal Volume (R3)": 6371000.7900,
    "GRS 80 semi-major axis (a)": 6378137.0,
    "GRS 80 semi-minor axis (b)": 6356752.314140,
    "Spherical Earth Approx. of Radius (RE)": 6366707.0195,
    "meridional radius of curvature at the equator": 6335439,
    "Maximum (the summit of Chimborazo)": 6384400,
    "Minimum (the floor of the Arctic Ocean)": 6352800,
    "Average distance from center to surface": 6371230,
}


class WGS84:
    A = 6378137.0  # Semi-major axis of the Earth in meters
    IF = 298.257223563  # Inverse flattening of the Earth

    F = 1.0 / IF  # flattening of the Earth
    B = A * (1 - F)  # Semi-minor axis of the Earth in meters
    E2 = 2 * F - F * F  # Squared eccentricity of the Earth

    @classmethod
    def geodetic2ecef(cls, coords: np.ndarray) -> np.ndarray:
        """Converts a list of geodetic coordinates (latitude, longitude) (in radians) to
            a list of ECEF coordinates (x, y, z).
        Ie. it converts from spherical coords to cartesian coords.

        See `pymap3d.ecef.geodetic2ecef` for an alternative implementation.
        """

        assert coords.ndim >= 2
        assert coords.shape[-1] == 2
        assert coords.dtype in (np.float32, np.float64)

        lat = coords[..., 0]
        lon = coords[..., 1]

        coslat = np.cos(lat)
        sinlat = np.sin(lat)

        normal = cls.A / (np.sqrt(1.0 - cls.E2 * (sinlat * sinlat)))

        x = normal * coslat * np.cos(lon)
        y = normal * coslat * np.sin(lon)
        z = normal * (1.0 - cls.E2) * sinlat

        return np.stack([x, y, z], axis=-1)

    @classmethod
    def ecef2geodetic(cls, coords: np.ndarray) -> np.ndarray:
        """Converts a list of ECEF coordinates (x, y, z) to
            a list of geodetic coordinates (latitude, longitude, altitude).
            Ie. it converts cartesian coords to from spherical coords.

        Based on `pymap3d/ecef.ecef2geodetic`.
        """

        assert coords.ndim >= 2
        assert coords.shape[-1] == 3
        assert coords.dtype in (np.float32, np.float64)

        x = coords[..., 0]
        y = coords[..., 1]
        z = coords[..., 2]

        r = np.sqrt(x * x + y * y + z * z)
        E = np.sqrt(cls.A**2 - cls.B**2)

        # eqn. 4a
        u = np.sqrt(0.5 * (r**2 - E**2) + 0.5 * np.sqrt((r**2 - E**2) ** 2 + 4 * E**2 * z**2))

        Q = np.hypot(x, y)
        huE = np.hypot(u, E)

        # eqn. 4b
        try:
            Beta = np.arctan(huE / u * z / np.hypot(x, y))
        except ZeroDivisionError:
            raise
            if z >= 0:
                Beta = np.pi / 2
            else:
                Beta = -np.pi / 2

        # eqn. 13
        eps = ((cls.B * u - cls.A * huE + E**2) * np.sin(Beta)) / (
            cls.A * huE * 1 / np.cos(Beta) - E**2 * np.cos(Beta)
        )

        Beta += eps

        lat = np.arctan(cls.A / cls.B * np.tan(Beta))
        lon = np.arctan2(y, x)

        # eqn. 7
        alt = np.hypot(z - cls.B * np.sin(Beta), Q - cls.A * np.cos(Beta))

        # inside ellipsoid?
        inside = x**2 / cls.A**2 + y**2 / cls.A**2 + z**2 / cls.B**2 < 1
        alt[inside] = -alt[inside]

        assert lat.shape == lon.shape == alt.shape
        return np.stack([lat, lon, alt], axis=-1)


def toint(s: str) -> Optional[int]:
    if s:
        return int(s)
    else:
        return None


def get_data(path: str, keep_dups: bool = False) -> Cities:
    """Read data from tsv file. Expect the following columns:

    cols = ("geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude", "feature class",
        "feature code", "country code", "cc2", "admin1 code", "admin2", "admin3", "admin4", "population",
        "elevation", "dem", "timezone", "modification date"
    )

    When keep_dups is False (the default) cities with the exact same coordinates as a previous one are skipped.
    This ensures more consistent results and should probably be fixed in the source data file.
    """

    with lzma.open(path, "rt", encoding="utf-8", newline="") as fr:
        lats = []
        lons = []
        cities = []

        known: Set[Tuple[float, float]] = set()
        for row in csv.reader(fr, delimiter="\t", quoting=csv.QUOTE_NONE):
            lat = float(row[4])
            lon = float(row[5])

            if not keep_dups:
                if (lat, lon) in known:
                    continue
                else:
                    known.add((lat, lon))

            city = City(
                row[1],
                row[6],
                row[7],
                row[8],
                row[10],
                row[11],
                row[12],
                row[13],
                int(row[14]),
                toint(row[15]),
                row[17],
            )

            lats.append(lat)
            lons.append(lon)
            cities.append(city)

    arr = np.array([lats, lons], dtype=np.float32).T
    arr = np.ascontiguousarray(np.deg2rad(arr))

    return Cities(arr, cities)


def _check_input(arr: np.ndarray, k: int, in_form: str, out_form: str) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != 2 or arr.dtype != np.float32:
        raise ValueError("Expected numpy array of radians with shape (x>0, 2) and dtype float32")

    if k < 1:
        raise ValueError(f"k must >= 1, not {k}")

    if in_form == out_form:
        pass
    elif in_form == "radians" and out_form == "degrees":
        arr = np.rad2deg(arr)
    elif in_form == "degrees" and out_form == "radians":
        arr = np.deg2rad(arr)
    elif in_form == "radians" and out_form == "ecef":
        arr = WGS84.geodetic2ecef(arr)
    elif in_form == "degrees" and out_form == "ecef":
        arr = np.deg2rad(arr)
        arr = WGS84.geodetic2ecef(arr)
    else:
        raise ValueError(f"Invalid input form: {in_form} or output form {out_form}")

    return arr


def _select_cities(cities: List[City], indices: np.ndarray) -> List[List[City]]:
    out = []
    for i in range(indices.shape[0]):
        out.append([cities[idx] for idx in indices[i]])
    return out


class ReverseGeocodeKdScipy:
    def __init__(self, path: Optional[str] = None) -> None:
        from scipy.spatial import KDTree

        if path is None:
            path = files(__package__).joinpath("data/cities1000.txt.xz")

        arr, self.cities = get_data(path)
        arr = WGS84.geodetic2ecef(arr)
        assert arr.dtype == np.float32
        self.tree = KDTree(arr)

    @overload
    def query(
        self, query_arr: np.ndarray, k: int, form: str, return_distance: Literal[True]
    ) -> Tuple[np.ndarray, np.ndarray, List[List[City]]]:
        ...

    @overload
    def query(
        self, query_arr: np.ndarray, k: int, form: str, return_distance: Literal[False]
    ) -> Tuple[np.ndarray, List[List[City]]]:
        ...

    def query(self, query_arr, k=2, form="radians", return_distance=True):
        query_arr = _check_input(query_arr, k, form, "ecef")

        distances, indices = self.tree.query(query_arr, k)
        if k == 1:
            distances = distances[..., None]
            indices = indices[..., None]
        cities = _select_cities(self.cities, indices)

        coords = self.tree.data[indices].astype(np.float32)
        coords = WGS84.ecef2geodetic(coords)
        coords = coords[..., :2]  # ignore altitude

        if return_distance:
            return coords, distances, cities
        else:
            return coords, cities


class ReverseGeocodeVpTreePython:
    """https://github.com/RickardSjogren/vptree"""

    @staticmethod
    def _euclidean(p1, p2):
        diff = p2[1] - p1[1]
        return np.sqrt(np.sum(diff * diff, axis=-1))

    def __init__(self, path: Optional[str] = None) -> None:
        from vptree import VPTree

        if path is None:
            path = files(__package__).joinpath("data/cities1000.txt.xz")

        arr, self.cities = get_data(path)
        arr = WGS84.geodetic2ecef(arr)
        assert arr.dtype == np.float32
        arr_with_index = [(i, v) for i, v in zip(range(len(arr)), arr)]
        self.tree = VPTree(arr_with_index, self._euclidean)

    def query(self, query_arr, k=2, form="radians", return_distance=True):
        query_arr = _check_input(query_arr, k, form, "ecef")

        indices = []
        coords = []
        distances = []

        for query in query_arr:
            query = (0, query)
            neighbors = self.tree.get_n_nearest_neighbors(query, k)
            _distances, pairs = zip(*neighbors)
            _indices, _coords = zip(*pairs)
            indices.append(_indices)
            coords.append(_coords)
            distances.append(_distances)

        indices = np.array(indices, dtype=np.int64)
        coords = np.array(coords, dtype=np.float32)
        distances = np.array(distances, dtype=np.float32)
        cities = _select_cities(self.cities, indices)

        coords = WGS84.ecef2geodetic(coords)
        coords = coords[..., :2]  # ignore altitude

        if return_distance:
            return coords, distances, cities
        else:
            return coords, cities


class ReverseGeocodeVpTreeSimd:
    """https://github.com/pablocael/pynear"""

    def __init__(self, path: Optional[str] = None) -> None:
        from pynear import VPTreeL2Index

        if path is None:
            path = files(__package__).joinpath("data/cities1000.txt.xz")

        arr, self.cities = get_data(path)
        self.arr = WGS84.geodetic2ecef(arr)
        assert arr.dtype == np.float32
        self.tree = VPTreeL2Index()
        self.tree.set(self.arr)

    @overload
    def query(
        self, query_arr: np.ndarray, k: int, form: str, return_distance: Literal[True]
    ) -> Tuple[np.ndarray, np.ndarray, List[List[City]]]:
        ...

    @overload
    def query(
        self, query_arr: np.ndarray, k: int, form: str, return_distance: Literal[False]
    ) -> Tuple[np.ndarray, List[List[City]]]:
        ...

    def query(self, query_arr, k=2, form="radians", return_distance=True):
        query_arr = _check_input(query_arr, k, form, "ecef")

        indices, distances = self.tree.searchKNN(query_arr, k)
        indices = np.array(indices, dtype=np.int64)[:, ::-1]
        distances = np.array(distances, dtype=np.float32)[:, ::-1]
        cities = _select_cities(self.cities, indices)

        coords = self.arr[indices]
        coords = WGS84.ecef2geodetic(coords)
        coords = coords[..., :2]  # ignore altitude

        if return_distance:
            return coords, distances, cities
        else:
            return coords, cities


class ReverseGeocodeKdLearn:
    def __init__(self, path: Optional[str] = None) -> None:
        from sklearn.neighbors import KDTree

        if path is None:
            path = files(__package__).joinpath("data/cities1000.txt.xz")

        arr, self.cities = get_data(path)
        arr = WGS84.geodetic2ecef(arr)
        assert arr.dtype == np.float32
        self.tree = KDTree(arr)  # copy is made here since input is float32 and float64 is needed
        # assert self.tree.data.base is arr

    @overload
    def query(
        self, query_arr: np.ndarray, k: int, form: str, return_distance: Literal[True]
    ) -> Tuple[np.ndarray, np.ndarray, List[List[City]]]:
        ...

    @overload
    def query(
        self, query_arr: np.ndarray, k: int, form: str, return_distance: Literal[False]
    ) -> Tuple[np.ndarray, List[List[City]]]:
        ...

    def query(self, query_arr, k=2, form="radians", return_distance=True):
        query_arr = _check_input(query_arr, k, form, "ecef")

        if return_distance:
            distances, indices = self.tree.query(query_arr, k, return_distance)
        else:
            indices = self.tree.query(query_arr, k, return_distance)

        cities = _select_cities(self.cities, indices)

        coords = np.asarray(self.tree.data)
        assert self.tree.data.base is coords.base.obj.base, "array was copied"
        coords = coords[indices].astype(np.float32)

        coords = WGS84.ecef2geodetic(coords)
        coords = coords[..., :2]  # ignore altitude

        if return_distance:
            return coords, distances, cities
        else:
            return coords, cities


class ReverseGeocodeBruteEuclidic:
    def __init__(self, path: Optional[str] = None) -> None:
        if path is None:
            path = files(__package__).joinpath("data/cities1000.txt.xz")

        arr, self.cities = get_data(path)
        self.arr = WGS84.geodetic2ecef(arr)

    @overload
    def query(
        self, query_arr: np.ndarray, k: int, form: str, return_distance: Literal[True]
    ) -> Tuple[np.ndarray, np.ndarray, List[List[City]]]:
        ...

    @overload
    def query(
        self, query_arr: np.ndarray, k: int, form: str, return_distance: Literal[False]
    ) -> Tuple[np.ndarray, List[List[City]]]:
        ...

    def query(self, query_arr, k=2, form="radians", return_distance=True):
        query_arr = _check_input(query_arr, k, form, "ecef")

        distances = euclidean_distances(query_arr, self.arr, squared=True)

        indices = np.argpartition(distances, range(k), axis=-1)[:, :k]
        cities = _select_cities(self.cities, indices)
        if return_distance:
            distances = np.take_along_axis(distances, indices, axis=-1)
            distances = np.sqrt(distances)  # distance matrix is squared

        coords = self.arr[indices]
        coords = WGS84.ecef2geodetic(coords)
        coords = coords[..., :2]  # ignore altitude

        if return_distance:
            return coords, distances, cities
        else:
            return coords, cities


class ReverseGeocodeBruteHaversine:
    radius = earth_radii["Spherical Earth Approx. of Radius (RE)"]  # see opt_geocoding.py

    def __init__(self, path: Optional[str] = None) -> None:
        if path is None:
            path = files(__package__).joinpath("data/cities1000.txt.xz")

        self.arr, self.cities = get_data(path)

    @overload
    def query(
        self, query_arr: np.ndarray, k: int, form: str, return_distance: Literal[True]
    ) -> Tuple[np.ndarray, np.ndarray, List[List[City]]]:
        ...

    @overload
    def query(
        self, query_arr: np.ndarray, k: int, form: str, return_distance: Literal[False]
    ) -> Tuple[np.ndarray, List[List[City]]]:
        ...

    def query(self, query_arr, k=2, form="radians", return_distance=True):
        query_arr = _check_input(query_arr, k, form, "radians")

        distances = haversine_distances(query_arr, self.arr)
        indices = np.argpartition(distances, range(k), axis=-1)[:, :k]
        cities = _select_cities(self.cities, indices)
        if return_distance:
            distances = np.take_along_axis(distances, indices, axis=-1)
            distances *= self.radius

        coords = self.arr[indices]

        if return_distance:
            return coords, distances, cities
        else:
            return coords, cities


class ReverseGeocodeBallHaversine:
    radius = earth_radii["Spherical Earth Approx. of Radius (RE)"]  # see opt_geocoding.py

    def __init__(self, path: Optional[str] = None) -> None:
        from sklearn.neighbors import BallTree

        if path is None:
            path = files(__package__).joinpath("data/cities1000.txt.xz")

        arr, self.cities = get_data(path)
        self.bt = BallTree(arr, metric="haversine")

    @overload
    def query(
        self, query_arr: np.ndarray, k: int, form: str, return_distance: Literal[True]
    ) -> Tuple[np.ndarray, np.ndarray, List[List[City]]]:
        ...

    @overload
    def query(
        self, query_arr: np.ndarray, k: int, form: str, return_distance: Literal[False]
    ) -> Tuple[np.ndarray, List[List[City]]]:
        ...

    def query(self, query_arr, k=2, form="radians", return_distance=True):
        query_arr = _check_input(query_arr, k, form, "radians")

        if return_distance:
            distances, indices = self.bt.query(query_arr, k, return_distance)
        else:
            indices = self.bt.query(query_arr, k, return_distance)
        cities = _select_cities(self.cities, indices)

        if return_distance:
            distances *= self.radius

        coords = np.asarray(self.bt.data)
        assert self.bt.data.base is coords.base.obj.base, "array was copied"
        coords = coords[indices]

        if return_distance:
            return coords, distances, cities
        else:
            return coords, cities

    def lat_lon(self, lat: float, lon: float, form: str) -> Tuple[List[float], float, City]:
        query_arr = np.array([[lat, lon]], dtype=np.float32)
        coords, distances, cities = self.query(query_arr, 1, form, True)
        return coords[0, 0].tolist(), distances[0, 0], cities[0][0]


ReverseGeocode = ReverseGeocodeBallHaversine
