import requests
from pathlib import Path
from utils import read_maps_api_key, compute_route, decimal_degrees_to_meters, tile_session_token
import config as cfg
from shapely.geometry import LineString, Point
import math
import logging
import os
import re
from PIL import Image
import time

class RouteTiler():
    def __init__(self, image_folder: str, zoom_level: int = 11, tilesize: int = 1024):
        if tilesize not in [256, 512, 1024]:
            raise ValueError("tilesize must be 256, 512 or 1024 pixels")
        self.api_key = read_maps_api_key()
        self.zoom_level = zoom_level
        self.tilesize = tilesize
        (cfg.imgpath / image_folder).mkdir(exist_ok=True)
        self.output_path = cfg.imgpath / image_folder


    @classmethod
    def get_scale_from_zoom(cls, zoom, lat):
        return 1 / (156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom))


    def _download_tile(self, session_token, x, y) -> tuple:
        url = f"https://tile.googleapis.com/v1/2dtiles/{self.zoom_level}/{x}/{y}"
        params = {
            "session": session_token,
            "key": self.api_key
        }

        response = requests.get(url, params=params)
        filepath = self.output_path / f'tile_x{x}_y{y}_z{self.zoom_level}.png'
        if response.status_code == 200:
            with open(filepath, "wb") as file:
                file.write(response.content)
            print(f"Tile at {x}, {y} downloaded successfully. Output path: {self.output_path}")
        else:
            print(f"Failed to download tile at {x}, {y}. Status code: {response.status_code}")
        return (x, y)


    def _lat_lng_to_tile_coordinate(self, lng, lat):
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
        logging.info(f"The longest distance between two sequential coordinates is: {longest_distance_meters:.2f}")
        return longest_distance_meters


    def make_image_from_route(self, origin : str, destination: str, **kwargs):
        route = compute_route(origin, destination, self.api_key, **kwargs)
        line = LineString(route['routes'][0]['polyline']['geoJsonLinestring']['coordinates'])
        logging.info(f"Found route, distance {route['routes'][0]['distanceMeters']} meters")
        scale = (1 / RouteTiler.get_scale_from_zoom(self.zoom_level, lat=line.coords[0][1]))*256
        logging.info(f"Tile scale at zoom level {self.zoom_level} is {scale} meters.")
        
        longest_distance_meters = self._max_interpoint_distance(line)
        if longest_distance_meters >= (scale)*0.9:
            logging.WARNING(f"Tile width shorter than interpoint distance, interpolating points.")
            line = self._interpolate_points(line)

        tilecoords = []
        tiles = self._calculate_intersecting_tiles(line)
        with tile_session_token(self.api_key, self.tilesize) as token:
            for i, tile in enumerate(tiles):
                logging.info(f"downloading tile {i} out of {len(tiles)}")
                (x, y) = self._download_tile(token, *tile)
                logging.info(f"tile at {(x, y)} successfully downloaded")
                tilecoords.append((x, y))
                time.sleep(1)

        self.stitch_tiles()


    def stitch_tiles(self):
        # Get a list of all PNG files in the tile directory
        tile_files = [f for f in os.listdir(self.output_path) if f.endswith('.png')]

        # Extract the tile coordinates from the file names using regex
        tiles = {}
        pattern = re.compile(r'tile_x(\d+)_y(\d+)_z(\d+)')
        for tile_file in tile_files:
            match = pattern.search(tile_file)
            if match:
                x, y, _ = map(int, match.groups())
                tiles[(x, y)] = Image.open(self.output_path / tile_file)

        # Find the dimensions of the final image
        max_x = max(x for x, _ in tiles.keys())
        min_x = min(x for x, _ in tiles.keys())
        max_y = max(y for _, y in tiles.keys())
        min_y = min(y for _, y in tiles.keys())

        width = (max_x-min_x + 1) * self.tilesize
        height = (max_y-min_y + 1) * self.tilesize

        # Create a new transparent image for the final result
        final_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))

        # Paste the tiles onto the final image
        for (x, y), tile in tiles.items():
            x_norm = x - min_x
            y_norm = y - min_y
            final_image.paste(tile, (x_norm * self.tilesize, y_norm * self.tilesize))

        # Save the final image as a PNG file
        final_image.save(self.output_path / 'final.png', 'PNG')


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
    rt = RouteTiler(image_folder='test_stitch', zoom_level=11, tilesize=1024)
    rt.make_image_from_route(origin="Fregattgatan 3, 21113 Malmö, Sweden",
                  destination="Stockholm, Sweden")

