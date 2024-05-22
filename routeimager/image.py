import requests
from pathlib import Path
from utils import read_maps_api_key, compute_route, decimal_degrees_to_meters, tile_session_token
import config as cfg
from shapely.geometry import LineString, Point
import math
import logging

class RouteTiler():
    def __init__(self, image_folder: str, zoom_level: int = 11, intended_dpi=200):
        self.api_key = read_maps_api_key()
        self.zoom_level = zoom_level
        self.tile_size = 256
        (cfg.imgpath / image_folder).mkdir(exist_ok=True)
        self.output_path = cfg.imgpath / image_folder

    @classmethod
    def get_scale_from_zoom(cls, zoom, lat):
        return 1 / (156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom))

    def _download_tile(self, session_token, x, y):
        url = f"https://tile.googleapis.com/v1/2dtiles/{zoom}/{x}/{y}"
        params = {
            "session": session_token,
            "key": self.api_key
        }

        response = requests.get(url, params=params)
        filepath = self.output_path / f'tile_{x}_{y}_{self.zoom_level}.png'
        if response.status_code == 200:
            with open(filepath, "wb") as file:
                file.write(response.content)
            print(f"Tile at {x}, {y} downloaded successfully. Output path: {output_path}")
        else:
            print(f"Failed to download tile at {x}, {y}. Status code: {response.status_code}")

    def _lat_lng_to_tile_coordinate(self, lat, lng):
        lat_rad = math.radians(lat)
        n = 2.0 ** self.zoom_level
        x = int((lng + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (x, y)

    def _max_interpoint_distance(self, line : LineString):
        distances = []
        for i in range(len(line.coords) - 1):
            start_coord = line.coords[i]
            end_coord = line.coords[i + 1]
            
            start_point = Point(start_coord)
            end_point = Point(end_coord)
            
            distance = start_point.distance(end_point)
            distances.append(distance)

        longest_distance = max(distances)
        longest_distance_meters = decimal_degrees_to_meters(longest_distance, line.coords[0][1])
        logging.INFO(f"The longest distance between two sequential coordinates is: {decimal_degrees_to_meters(longest_distance, line.coords[0][1])}")
        return longest_distance_meters

    def make_image_from_route(self, origin : str, destination: str, **kwargs):
        route = compute_route(origin, destination, self.api_key, **kwargs)
        line = LineString(route['routes'][0]['polyline']['geoJsonLinestring']['coordinates'])
        logging.info(f"Found route, distance {route['routes'][0]['distanceMeters']} meters")
        scale = (1 / RouteTiler.get_scale_from_zoom(self.zoom_level, lat=line.coords[0][1]))*self.tile_size
        logging.info(f"Tile scale at zoom level {self.zoom_level} is {scale} meters.")
        
        longest_distance_meters = self._max_interpoint_distance(line)
        if longest_distance_meters >= (scale)*0.9:
            logging.WARNING(f"Tile width shorter than interpoint distance, interpolating points.")
            line = self._interpolate_points(line)

        tiles = self._calculate_intersecting_tiles(line)
        with tile_session_token(self.api_key) as token:
            for tile in tiles:
                self._download_tile(token, *tile)

    def _calculate_intersecting_tiles(self, line):
        tiles = []
        for coord in line.coords:
            tiles.append(self._lat_lng_to_tile_coordinate(*coord))

        tiles = list(set(tiles))
        return tiles


    def _interpolate_points(self, line: LineString):
        raise NotImplementedError




if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    rt = RouteTiler(image_folder='test')
    rt.make_image(origin="Fregattgatan 3, 21113 Malmö, Sweden",
                  destination="Stockholm, Sweden")

