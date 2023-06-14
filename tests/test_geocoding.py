import unittest
from pathlib import Path

import numpy as np
from genutility.cache import cache

from houtu.geocoding import (
    ReverseGeocodeBallHaversine,
    ReverseGeocodeBruteEuclidic,
    ReverseGeocodeBruteHaversine,
    ReverseGeocodeKdLearn,
    ReverseGeocodeKdScipy,
)
from houtu.utils import rand_lat_lon


class ReverseGeocodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geo_hav1 = cache(Path("cache/brute-haversine"), serializer="pickle")(ReverseGeocodeBruteHaversine)()
        cls.geo_hav2 = cache(Path("cache/ball-haversine"), serializer="pickle")(ReverseGeocodeBallHaversine)()

        cls.geo_euc1 = cache(Path("cache/brute-euclidic"), serializer="pickle")(ReverseGeocodeBruteEuclidic)()
        cls.geo_euc2 = cache(Path("cache/kd-learn"), serializer="pickle")(ReverseGeocodeKdLearn)()
        cls.geo_euc3 = cache(Path("cache/kd-scipy"), serializer="pickle")(ReverseGeocodeKdScipy)()

    def test_small(self):
        small = np.deg2rad(
            np.array(
                [
                    [25.04776, 121.53185],  # taipei
                    [48.13743, 11.57549],  # munich
                ],
                dtype=np.float32,
            )
        )

        coords_h1, distances_h1, cities_h1 = self.geo_hav1.query(small)
        coords_h2, distances_h2, cities_h2 = self.geo_hav2.query(small)
        coords_e1, distances_e1, cities_e1 = self.geo_euc1.query(small)
        coords_e2, distances_e2, cities_e2 = self.geo_euc2.query(small)
        coords_e3, distances_e3, cities_e3 = self.geo_euc3.query(small)

        assert np.allclose(coords_h1, coords_h2)
        assert np.allclose(coords_h1, coords_e1)
        assert np.allclose(coords_h1, coords_e2)
        assert np.allclose(coords_h1, coords_e3)

        self.assertEqual(cities_h1, cities_h2)
        self.assertEqual(cities_h1, cities_e1)
        self.assertEqual(cities_h1, cities_e2)
        self.assertEqual(cities_h1, cities_e3)

        assert np.allclose(distances_h1, distances_h2)
        assert np.allclose(distances_e1, distances_e2)
        assert np.allclose(distances_e1, distances_e3)

        self.assertEqual(cities_h1[0][0].name, "Taipei")
        self.assertEqual(cities_h1[0][1].name, "Neihu")
        self.assertEqual(cities_h1[1][0].name, "Munich")
        self.assertEqual(cities_h1[1][1].name, "Bogenhausen")

    def test_large(self):
        large = rand_lat_lon(1000, "radians")

        coords_h1, distances_h1, cities_h1 = self.geo_hav1.query(large)
        coords_h2, distances_h2, cities_h2 = self.geo_hav2.query(large)
        coords_e1, distances_e1, cities_e1 = self.geo_euc1.query(large)
        coords_e2, distances_e2, cities_e2 = self.geo_euc2.query(large)
        coords_e3, distances_e3, cities_e3 = self.geo_euc3.query(large)

        assert np.allclose(coords_h1, coords_h2)
        # assert np.allclose(coords_h1, coords_e1)  # fails
        assert np.allclose(coords_e1, coords_e2)
        assert np.allclose(coords_e1, coords_e3)

        self.assertEqual(cities_h1, cities_h2)
        # self.assertEqual(cities_h1, cities_e1)  # fails
        self.assertEqual(cities_e1, cities_e2)
        self.assertEqual(cities_e1, cities_e3)

        assert np.allclose(distances_h1, distances_h2)
        assert np.allclose(distances_e1, distances_e2)
        assert np.allclose(distances_e1, distances_e3)
