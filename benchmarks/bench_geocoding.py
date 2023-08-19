from timeit import repeat

from houtu.geocoding import (
    ReverseGeocodeBallHaversine,
    ReverseGeocodeBruteEuclidic,
    ReverseGeocodeBruteHaversine,
    ReverseGeocodeKdLearn,
    ReverseGeocodeKdScipy,
    ReverseGeocodeVpTreePython,
    ReverseGeocodeVpTreeSimd,
)
from houtu.utils import rand_lat_lon


def benchmark():
    geo_h1 = ReverseGeocodeBruteHaversine()
    geo_h2 = ReverseGeocodeBallHaversine()
    geo_e1 = ReverseGeocodeBruteEuclidic()
    geo_e2 = ReverseGeocodeKdLearn()
    geo_e3 = ReverseGeocodeKdScipy()
    geo_e4 = ReverseGeocodeVpTreeSimd()
    geo_e5 = ReverseGeocodeVpTreePython()

    geos = [geo_h1, geo_h2, geo_e1, geo_e2, geo_e3, geo_e4, geo_e5]

    query = rand_lat_lon(2, "radians")
    for geo in geos:
        name = geo.__class__.__name__
        try:
            seconds = min(repeat("geo.query(query)", globals={"geo": geo, "query": query}, number=1000))
        except MemoryError:
            seconds = "MemoryError"
        print(name, seconds)

    query = rand_lat_lon(100, "radians")
    for geo in geos:
        name = geo.__class__.__name__
        try:
            seconds = min(repeat("geo.query(query)", globals={"geo": geo, "query": query}, number=100))
        except MemoryError:
            seconds = "MemoryError"
        print(name, seconds)


if __name__ == "__main__":
    benchmark()
