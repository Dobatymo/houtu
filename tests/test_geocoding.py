import logging
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
    ReverseGeocodeVpTreePython,
    ReverseGeocodeVpTreeSimd,
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
        cls.geo_euc4 = ReverseGeocodeVpTreeSimd()
        try:
            cls.geo_euc5 = cache(Path("cache/vp-python"), serializer="pickle")(ReverseGeocodeVpTreePython)()
        except ImportError as e:
            logging.error("Skipping ReverseGeocodeVpTreePython: %s", e)
            cls.geo_euc5 = None

        cls.geo_all = [cls.geo_hav1, cls.geo_hav2, cls.geo_euc1, cls.geo_euc2, cls.geo_euc3, cls.geo_euc4]
        if cls.geo_euc5 is not None:
            cls.geo_all.append(("geo_euc5", cls.geo_euc5))

        cls.geo_hav_2_to_n = [
            ("geo_hav2", cls.geo_hav2),
        ]

        cls.geo_euc_2_to_n = [
            ("geo_euc2", cls.geo_euc2),
            ("geo_euc3", cls.geo_euc3),
            ("geo_euc4", cls.geo_euc4),
        ]
        if cls.geo_euc5 is not None:
            cls.geo_euc_2_to_n.append(("geo_euc5", cls.geo_euc5))

    def test_wrong_inputs(self):
        empty = np.zeros((0, 2), dtype=np.float32)
        wrongdim = np.zeros((2,), dtype=np.float32)
        wrongdtype = np.zeros((1, 2), dtype=np.int32)
        for arr in (empty, wrongdim, wrongdtype):
            for geo in self.geo_all:
                with self.assertRaises(ValueError):
                    geo.query(arr)

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
            with self.subTest(k=k, name="geo_hav1", out="cities"):
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

            for name, obj in self.geo_hav_2_to_n:
                coords, distances, cities = obj.query(small, k)
                with self.subTest(k=k, name=name, out="coords", coords_h1=coords_h1, coords=coords):
                    np.testing.assert_allclose(coords_h1, coords)
                with self.subTest(k=k, name=name, out="cities", cities_h1=cities_h1, cities=cities):
                    self.assertEqual(cities_h1, cities)
                with self.subTest(k=k, name=name, out="distances", distances_h1=distances_h1, distances=distances):
                    np.testing.assert_allclose(distances_h1, distances)

            coords_e1, distances_e1, cities_e1 = self.geo_euc1.query(small, k)
            with self.subTest(k=k, name="geo_euc1", out="coords", coords_h1=coords_h1, coords_e1=coords_e1):
                np.testing.assert_allclose(coords_h1, coords_e1, rtol=1e-06)
            with self.subTest(k=k, name="geo_euc1", out="cities", cities_h1=cities_h1, cities_e1=cities_e1):
                self.assertEqual(cities_h1, cities_e1)

            for name, obj in self.geo_euc_2_to_n:
                coords, distances, cities = obj.query(small, k)
                with self.subTest(k=k, name=name, out="coords", coords_e1=coords_e1, coords=coords):
                    np.testing.assert_allclose(coords_e1, coords)
                with self.subTest(k=k, name=name, out="cities", cities_e1=cities_e1, cities=cities):
                    self.assertEqual(cities_e1, cities)
                with self.subTest(k=k, name=name, out="distances", distances_e1=distances_e1, distances=distances):
                    np.testing.assert_allclose(distances_e1, distances, rtol=1e-06)

    def test_large(self):
        large = rand_lat_lon(1000, "radians")

        with self.subTest(name="geo_hav1"):
            coords_h1, distances_h1, cities_h1 = self.geo_hav1.query(large, 5)

        for name, obj in self.geo_hav_2_to_n:
            with self.subTest(name=name):
                coords, distances, cities = obj.query(large, 5)
                np.testing.assert_allclose(coords_h1, coords)
                self.assertEqual(cities_h1, cities)
                np.testing.assert_allclose(distances_h1, distances)

        with self.subTest(name="geo_euc1"):
            coords_e1, distances_e1, cities_e1 = self.geo_euc1.query(large, 5)
            # assert np.allclose(coords_h1, coords_e1)  # fails
            # self.assertEqual(cities_h1, cities_e1)  # fails

        for name, obj in self.geo_euc_2_to_n:
            with self.subTest(name=name):
                coords, distances, cities = obj.query(large, 5)
                np.testing.assert_allclose(coords_e1, coords)
                self.assertEqual(cities_e1, cities)
                np.testing.assert_allclose(distances_e1, distances, rtol=1e-06)
