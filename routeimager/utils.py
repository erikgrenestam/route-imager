import urllib.parse
import requests
import config as cfg
import math
from contextlib import contextmanager
import json

def compute_route(origin: str, 
                  destination: str,
                  api_key: str, 
                  travel_mode='DRIVE', 
                  polyline_encoding='GEO_JSON_LINESTRING',
                  polyline_quality='HIGH_QUALITY'):
    url = 'https://routes.googleapis.com/directions/v2:computeRoutes'
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': api_key,
        'X-Goog-FieldMask': 'routes.duration,routes.distanceMeters,routes.polyline'
    }
    data = {
        'origin': {
            'address': origin
        },
        'destination': {
            'address': destination
        },
        'travelMode': travel_mode,
        'polylineEncoding': polyline_encoding,
        'polylineQuality': polyline_quality
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()
    else:
        print(f'Request failed with status code: {response.status_code}')
        return None

def read_maps_api_key(path=(cfg.root / 'credentials.txt')):
    with open(path, 'r') as file:
        api_key = file.read().strip()
    return api_key

@contextmanager
def tile_session_token(api_key, map_type="satellite", language="en-US", region="us", highDpi=True,
                  layer_types=None, overlay=False, scale="scaleFactor4x", styles=None):
    url = f"https://tile.googleapis.com/v1/createSession?key={api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "mapType": map_type,
        "language": language,
        "region": region,
        "overlay": overlay,
        "scale": scale,
        "highDpi": highDpi
    }

    if layer_types:
        data["layerTypes"] = layer_types
    if styles:
        data["styles"] = styles

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        session_token = response.json().get("session")
        yield session_token
    else:
        raise Exception(f"Request failed with status code: {response.status_code}")

def geocode_address(address, api_key=None):
    if api_key is None:
        api_key = read_maps_api_key(cfg.root / 'credentials.txt')
    # URL-encode the address
    encoded_address = urllib.parse.quote(address)

    # Construct the API request URL
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={encoded_address}&key={api_key}"

    try:
        # Send a GET request to the Geocoding API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

            # Check if the response contains any results
            if data["status"] == "OK" and len(data["results"]) > 0:
                # Extract the latitude and longitude from the first result
                return data
            else:
                print("No results found for the given address.")
                return None
        else:
            print(f"Request failed with status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the request: {e}")
        return None
    
def decimal_degrees_to_meters(decimal_degrees, latitude):
    """
    Converts decimal degrees to meters based on the latitude.
    
    Parameters:
    - decimal_degrees: The decimal degrees value to be converted.
    - latitude: The latitude in decimal degrees.
    
    Returns:
    - The equivalent distance in meters.
    """
    # Earth's radius in meters
    earth_radius = 6371000
    
    # Convert latitude to radians
    lat_rad = math.radians(latitude)
    
    # Calculate the conversion factors
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
    m_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)
    
    # Calculate the distance in meters
    distance_lat = decimal_degrees * m_per_deg_lat
    distance_lon = decimal_degrees * m_per_deg_lon
    
    # Calculate the total distance using the Pythagorean theorem
    distance = math.sqrt(distance_lat**2 + distance_lon**2)
    
    return distance