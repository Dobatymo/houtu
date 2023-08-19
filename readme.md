# houtu

Offline fast reverse geocoding. Named after the Chinese Goddess of the Earth *Houtu*.

Various implementations are included.
- Naive computation of haversine distance matrix. Coordinates are not converted.
- Using ball-tree with haversine metric. Coordinates are not converted.
- Naive computation of Euclidean distance matrix. Coordinates are converted from geodetic to ECEF and back.
- Using kd-tree with Euclidean metric. Coordinates are converted from geodetic to ECEF and back.

## Install

- requires Python 3.8+
- run `pip install houtu`

## Use

```python
import numpy as np
from houtu import ReverseGeocode
rg = ReverseGeocode()
arr = np.array([
    [25.04776, 121.53185],  # taipei
    [48.13743, 11.57549],  # munich
], dtype=np.float32)
coords, distances, cities = rg.query(arr, k=2, form="degrees")
print(cities)
```

### Output

```
[[
  City(name='Taipei', feature_class='P', feature_code='PPLC', country_code='TW', admin1_code='04', admin2_code='TPE', admin3_code='', admin4_code='', population=7871900, elevation=None, timezone='Asia/Taipei'),
  City(name='Neihu', feature_class='P', feature_code='PPL', country_code='TW', admin1_code='04', admin2_code='TPE', admin3_code='A14', admin4_code='63000100010', population=271594, elevation=None, timezone='Asia/Taipei')
], [
  City(name='Munich', feature_class='P', feature_code='PPLA', country_code='DE', admin1_code='02', admin2_code='091', admin3_code='09162', admin4_code='09162000', population=1260391, elevation=None, timezone='Europe/Berlin'),
  City(name='Bogenhausen', feature_class='P', feature_code='PPLX', country_code='DE', admin1_code='02', admin2_code='091', admin3_code='09162', admin4_code='09162000', population=77542, elevation=None, timezone='Europe/Berlin')
]]
```

## Benchmark

### Batch of 2

| class | time/s |
| ----- | ------ |
|ReverseGeocodeBruteHaversine | 35.674 |
|ReverseGeocodeBallHaversine | 0.192 |
|ReverseGeocodeBruteEuclidic | 22.110 |
|ReverseGeocodeKdLearn | 0.392 |
|ReverseGeocodeKdScipy | 0.317 |

### Batch of 100

| class | time/s |
| ----- | ------ |
|ReverseGeocodeBruteHaversine | 193.119 |
|ReverseGeocodeBallHaversine | 0.918 |
|ReverseGeocodeBruteEuclidic | 86.881 |
|ReverseGeocodeKdLearn | 0.122 |
|ReverseGeocodeKdScipy | 0.095 |

## Optimize

```
vincenty haversine
mean_absolute_error IUGG radius of sphere of same volume (R3) 897.7743386541117
mean_relative_error Spherical Earth Approx. of Radius (RE) 0.0018409171841267883
geodesic haversine
mean_absolute_error IUGG radius of sphere of same volume (R3) 897.7743386541117
mean_relative_error Spherical Earth Approx. of Radius (RE) 0.0018409171841267883
vincenty euclidic {'mean_absolute_error': 976.6828669488417, 'mean_relative_error': 0.0006505034968118167}
geodesic euclidic {'mean_absolute_error': 976.628990753796, 'mean_relative_error': 0.0006465253527447903}
```

## Development

- run tests `python -m unittest discover -s tests`

## Sources

Cities database file `houtu/data/cities1000.txt.xz` is downloaded and recompressed from <http://download.geonames.org/export/dump/cities1000.zip> (Creative Commons Attribution 4.0 License).

### GeoNames issues

There are about 50 pairs of cities which have the exact same coordinates, for example `Fajã de Baixo` and `Rosto de Cão`.
