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

    def test_wrong_inputs(self):
        empty = np.zeros((0, 2), dtype=np.float32)
        wrongdim = np.zeros((2,), dtype=np.float32)
        wrongdtype = np.zeros((1, 2), dtype=np.int32)
        for arr in (empty, wrongdim, wrongdtype):
            with self.assertRaises(ValueError):
                coords_h1, distances_h1, cities_h1 = self.geo_hav1.query(arr)
            with self.assertRaises(ValueError):
                coords_h2, distances_h2, cities_h2 = self.geo_hav2.query(arr)
            with self.assertRaises(ValueError):
                coords_e1, distances_e1, cities_e1 = self.geo_euc1.query(arr)
            with self.assertRaises(ValueError):
                coords_e2, distances_e2, cities_e2 = self.geo_euc2.query(arr)
            with self.assertRaises(ValueError):
                coords_e3, distances_e3, cities_e3 = self.geo_euc3.query(arr)

    def test_small(self):
        small = np.deg2rad(
            np.array(
                [
                    [25.04776, 121.53185],  # taipei
                    [48.13743, 11.57549],  # munich
                    [40.71427, -74.00597],  # new york city
                ],
                dtype=np.float32,
            )
        )

        for k in (1, 2, 3, 4, 5):
            coords_h1, distances_h1, cities_h1 = self.geo_hav1.query(small, k)
            coords_h2, distances_h2, cities_h2 = self.geo_hav2.query(small, k)
            coords_e1, distances_e1, cities_e1 = self.geo_euc1.query(small, k)
            coords_e2, distances_e2, cities_e2 = self.geo_euc2.query(small, k)
            coords_e3, distances_e3, cities_e3 = self.geo_euc3.query(small, k)

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

            if k >= 1:
                self.assertEqual(cities_h1[0][0].name, "Taipei")
                self.assertEqual(cities_h1[1][0].name, "Munich")
                self.assertEqual(cities_h1[2][0].name, "New York City")
            if k >= 2:
                self.assertEqual(cities_h1[0][1].name, "Neihu")
                self.assertEqual(cities_h1[1][1].name, "Bogenhausen")
                self.assertEqual(cities_h1[2][1].name, "Financial District")
            if k >= 3:
                self.assertEqual(cities_h1[0][2].name, "Banqiao")
                self.assertEqual(cities_h1[1][2].name, "Unterf\xf6hring")
                self.assertEqual(cities_h1[2][2].name, "Chinatown")
            if k >= 4:
                self.assertEqual(cities_h1[0][3].name, "Xindian")
                self.assertEqual(cities_h1[1][3].name, "Unterhaching")
                self.assertEqual(cities_h1[2][3].name, "East Village")
            if k >= 5:
                self.assertEqual(cities_h1[0][4].name, "Shulin")
                self.assertEqual(cities_h1[1][4].name, "Pasing")
                self.assertEqual(cities_h1[2][4].name, "Brooklyn Heights")

    def test_large(self):
        large = rand_lat_lon(1000, "radians")

        coords_h1, distances_h1, cities_h1 = self.geo_hav1.query(large, 5)
        coords_h2, distances_h2, cities_h2 = self.geo_hav2.query(large, 5)
        coords_e1, distances_e1, cities_e1 = self.geo_euc1.query(large, 5)
        coords_e2, distances_e2, cities_e2 = self.geo_euc2.query(large, 5)
        coords_e3, distances_e3, cities_e3 = self.geo_euc3.query(large, 5)

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
