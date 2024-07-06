import requests
from pathlib import Path
from utils import read_maps_api_key, compute_route, decimal_degrees_to_meters, tile_session_token, meters_per_degree_lat_lon
import config as cfg
from shapely.geometry import LineString, Point, box
from shapely.ops import transform
from shapely import simplify
import pyproj
import math
import logging
import os
import re
from PIL import Image
import time

class RouteTiler():
    def __init__(self, 
                 image_folder: str, 
                 buffer_crs: str = None, 
                 zoom_level: int = 11, 
                 tilesize: int = 1024,
                 buffer_distance: int = 4000, 
                 wide_route: bool = False):
        if tilesize not in [256, 512, 1024]:
            raise ValueError("tilesize must be 256, 512 or 1024 pixels")
        self.api_key = read_maps_api_key()
        self.zoom_level = zoom_level
        self.tilesize = tilesize
        self.buffer_distance = buffer_distance
        (cfg.imgpath / image_folder).mkdir(exist_ok=True)
        self.output_path = cfg.imgpath / image_folder
        self.wide_route = wide_route
        self.route_line = None
        self.buffer_polygon = None
        self.wgs_bbox = None
        self.bbox_dict = None
        if buffer_crs is not None:
            self.project = pyproj.Transformer.from_crs(pyproj.CRS("EPSG:4326"), 
                                                    pyproj.CRS(buffer_crs), 
                                                    always_xy=True).transform
            self.unproject = pyproj.Transformer.from_crs(pyproj.CRS(buffer_crs), 
                                                        pyproj.CRS("EPSG:4326"), 
                                                        always_xy=True).transform
        else:
            self.project = None
            self.unproject = None


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
    

    def _tile_coordinate_to_wgs_bounds(self, x, y):
        n = 2.0 ** self.zoom_level
        lon_deg_left = x / n * 360.0 - 180.0
        lon_deg_right = (x + 1) / n * 360.0 - 180.0
        lat_rad_top = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat_rad_bottom = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
        lat_deg_top = math.degrees(lat_rad_top)
        lat_deg_bottom = math.degrees(lat_rad_bottom)
        #left_bottom_proj = self.project(lon_deg_left, lat_deg_bottom)
        #right_top_proj = self.project(lon_deg_right, lat_deg_top)
        return (lon_deg_left, lat_deg_bottom, lon_deg_right, lat_deg_top)


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

    # Function to convert CRS to pixel coordinates    
    def _wgs84_to_pixel(self, lon, lat):
        tile = self._lat_lng_to_tile_coordinate(lon, lat)
        tile_x_idx = tile[0] - self.min_tile_x
        tile_y_idx = self.max_tile_y - tile[1]
        bbox = self._tile_coordinate_to_wgs_bounds(*tile)

        #lat_len, lon_len = meters_per_degree_lat_lon(math.radians((bbox[1] + bbox[3]) / 2))

        # Calculate the physical dimensions of the bounding box
        #lat_diff_meters = (bbox[3] - bbox[1]) * lat_len
        #lon_diff_meters = (bbox[2] - bbox[0]) * lon_len
        
        # Calculate scaling factors
        x_scale = self.tilesize / (bbox[2] - bbox[0])
        y_scale = self.tilesize / (bbox[3] - bbox[1])

        # Convert to pixel coordinates
        pixel_x = (lon - bbox[0]) * x_scale + tile_x_idx*self.tilesize
        pixel_y = (lat - bbox[1]) * y_scale + tile_y_idx*self.tilesize  

        return int(pixel_x), int(pixel_y)


    def make_image_from_route(self, origin : str, destination: str, download: bool = False, **kwargs):
        route = compute_route(origin, destination, self.api_key, **kwargs)
        route_line = LineString(route['routes'][0]['polyline']['geoJsonLinestring']['coordinates'])
        self.route_line = simplify(route_line, 0.0005)

        # Create transformation functions
        if self.project is not None:
            projected_linestring = transform(self.project, self.route_line)
            self.buffer_polygon = projected_linestring.buffer(self.buffer_distance)
            self.buffer_polygon = simplify(self.buffer_polygon, self.buffer_distance*0.025, preserve_topology=True)
            self.buffer_polygon = transform(self.unproject, self.buffer_polygon)
        else:
            self.buffer_polygon = self.route_line.buffer(self.buffer_distance)
            self.buffer_polygon = simplify(self.buffer_polygon, self.buffer_distance*0.025, preserve_topology=True)

        logging.info(f"Found route, distance {route['routes'][0]['distanceMeters']} meters")
        scale = (1 / RouteTiler.get_scale_from_zoom(self.zoom_level, lat=self.route_line.coords[0][1]))*256
        logging.info(f"Tile scale at zoom level {self.zoom_level} is {scale} meters.")
        
        longest_distance_meters = self._max_interpoint_distance(self.route_line)
        if longest_distance_meters >= (scale)*0.9:
            logging.warning(f"Tile width shorter than interpoint distance, interpolating points.")
            self.route_line = self._interpolate_points(self.route_line, (scale)*0.5)

        tilecoords = []
        tiles = self._calculate_intersecting_tiles(self.route_line)

        if download:
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
        tile_png_files = [f for f in os.listdir(self.output_path) if f.endswith('.png')]

        # Save each tile as a JPEG
        for file in tile_png_files:
            png = Image.open(self.output_path / file)
            rgb_png = png.convert("RGB")
            rgb_png.save(self.output_path / f"{file.replace('.png', '.jpeg')}", "JPEG", optimize=True, quality=95)

        tile_jpeg_files = [f for f in os.listdir(self.output_path) if f.endswith('.jpeg')]

        # Extract the tile coordinates from the file names using regex
        tiles = {}
        pattern = re.compile(r'tile_x(\d+)_y(\d+)_z(\d+)')
        for tile_file in tile_jpeg_files:
            match = pattern.search(tile_file)
            if match:
                x, y, _ = map(int, match.groups())
                tiles[(x, y)] = Image.open(self.output_path / tile_file)

        # Find the dimensions of the final image
        self.max_tile_x = max(x for x, _ in tiles.keys())
        self.min_tile_x = min(x for x, _ in tiles.keys())
        self.max_tile_y = max(y for _, y in tiles.keys())
        self.min_tile_y = min(y for _, y in tiles.keys())

        width = (self.max_tile_x - self.min_tile_x + 1) * self.tilesize
        height = (self.max_tile_y - self.min_tile_y + 1) * self.tilesize

        pixel_coords = [self._wgs84_to_pixel(x, y) for x, y in self.buffer_polygon.exterior.coords]

        svg_polygon = '<polygon points="{}" style="fill:red;stroke:red;stroke-width:1" />'.format(
            ' '.join([f'{int(x)},{height-int(y)}' for x, y in pixel_coords]) # Invert y-axis for pixel coordinates
        )
        svg_content = f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" version="1.1">{svg_polygon}</svg>'

        with open(self.output_path / 'buffer_polygon.svg', 'w') as file:
            file.write(svg_content)

        # Create a new white image for the final result
        if width > 65500 or height > 65500:
            x_splits = (width // 65500) + 1
            y_splits = (height // 65500) + 1
            split_width = width // x_splits
            split_height = height // y_splits
            split_images = [[Image.new('RGB', (split_width, split_height), (255, 255, 255)) for _ in range(y_splits)] for _ in range(x_splits)]
            
            for (x, y), tile in tiles.items():
                x_norm = x - self.min_tile_x
                y_norm = y - self.min_tile_y
                x_split_index = x_norm // (split_width // self.tilesize)
                y_split_index = y_norm // (split_height // self.tilesize)
                x_offset = (x_norm % (split_width // self.tilesize)) * self.tilesize
                y_offset = (y_norm % (split_height // self.tilesize)) * self.tilesize
                split_images[x_split_index][y_split_index].paste(tile, (x_offset, y_offset))
            
            for x_idx, row in enumerate(split_images):
                for y_idx, img in enumerate(row):
                    img.save(self.output_path / f'final_part_x{x_idx + 1}_y{y_idx + 1}.jpeg', 'JPEG', optimize=True, quality=95)

        else:
            final_image = Image.new('RGB', (width, height), (255, 255, 255))

            # Paste the tiles onto the final image
            for (x, y), tile in tiles.items():
                x_norm = x - min_tile_x
                y_norm = y - min_tile_y
                final_image.paste(tile, (x_norm * self.tilesize, y_norm * self.tilesize))

            # Save the final image as a JPEG file
            final_image.save(self.output_path / 'final.jpeg', 'JPEG', optimize=True, quality=95)


    def _calculate_intersecting_tiles(self, line):
        def find_enclosing_bbox(coord, bounding_boxes):
            point = Point(coord)
            for bbox in bounding_boxes:
                if box(*bbox).contains(point):
                    return bbox
            return None
        
        tiles = set()
        lon_deg_left, lat_deg_bottom, lon_deg_right, lat_deg_top = None, None, None, None

        for coord in line.coords:
            tile = self._lat_lng_to_tile_coordinate(*coord)
            tiles.add(tile)
            if self.wide_route:
                adj_tiles = self._get_adjacent_tiles(*tile)
                tiles.update(adj_tiles)

        return list(tiles)
    

    def _get_adjacent_tiles(self, x, y):
        adjacent_tiles = [
            (x - 1, y - 1), (x, y - 1), (x + 1, y - 1),  # Top row
            (x - 1, y),                 (x + 1, y),      # Middle row
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)   # Bottom row
        ]
        return adjacent_tiles

    def _interpolate_points(self, line: LineString, max_distance):
        """
        Interpolates points along the given LineString to ensure that the maximum distance
        between any two consecutive points is less than max_distance meters.

        Args:
            line (LineString): The original LineString.
            max_distance (float): The maximum allowed distance between consecutive points in meters.

        Returns:
            LineString: A new LineString with interpolated points.
        """
        def haversine_distance(coord1, coord2):
            # Calculate the Haversine distance between two points on the Earth
            lon1, lat1 = coord1
            lon2, lat2 = coord2
            R = 6371000  # Radius of the Earth in meters
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            delta_phi = math.radians(lat2 - lat1)
            delta_lambda = math.radians(lon2 - lon1)

            a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            return R * c

        new_coords = []
        for i in range(len(line.coords) - 1):
            start_coord = line.coords[i]
            end_coord = line.coords[i + 1]
            new_coords.append(start_coord)

            distance = haversine_distance(start_coord, end_coord)
            num_intermediate_points = int(distance // max_distance)

            for j in range(1, num_intermediate_points + 1):
                intermediate_point = (
                    start_coord[0] + (end_coord[0] - start_coord[0]) * j / (num_intermediate_points + 1),
                    start_coord[1] + (end_coord[1] - start_coord[1]) * j / (num_intermediate_points + 1)
                )
                new_coords.append(intermediate_point)

        new_coords.append(line.coords[-1])  # Add the last coordinate
        logging.info(f"Points along line interpolated. Original length: {len(line.coords)}, new line: {len(new_coords)}")
        return LineString(new_coords)




if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    rt = RouteTiler(image_folder='test_stitch_12', zoom_level=12, tilesize=1024, wide_route=True)
    rt.make_image_from_route(origin="Fregattgatan 3, 21113 Malmö, Sweden",
                  destination="Stockholm, Sweden", download=False)

