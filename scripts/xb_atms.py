#!/usr/bin/env python3
"""
Script to fetch Xalq Bank ATM locations and save to CSV
"""

import requests
import csv
import json
from typing import List, Dict
import os


def fetch_service_network_data(url: str) -> Dict:
    """
    Fetch service network data from Xalq Bank API

    Args:
        url: API endpoint URL

    Returns:
        API response dictionary
    """
    print(f"Fetching data from {url}...")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'X-Requested-With': 'XMLHttpRequest'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.json()
    print(f"Successfully fetched data")

    return data


def extract_atm_locations(data: Dict) -> List[Dict]:
    """
    Extract ATM locations from API response

    Args:
        data: API response dictionary

    Returns:
        List of ATM location dictionaries
    """
    atm_locations = []

    # Navigate: data -> blocks[0] -> blocks
    api_data = data.get('data', {})
    blocks = api_data.get('blocks', [])

    if blocks:
        # Get the first block which contains the branches data
        branches_block = blocks[0]
        locations = branches_block.get('blocks', [])

        for location in locations:
            # Check if this is an ATM (category_name = "ATM-lər")
            category = location.get('category', {})
            if category.get('category_name') == 'ATM-lər':
                # Extract coordinates
                coords = location.get('coordinates', {})
                lat = coords.get('latitude')
                lon = coords.get('longitude')

                # Extract working hours
                working_days = location.get('working_days', [])
                work_hours = working_days[0].get('value', '') if working_days else ''

                atm_locations.append({
                    'id': location.get('id'),
                    'title': location.get('title'),
                    'slug': location.get('slug'),
                    'address': location.get('address'),
                    'phone': location.get('phone'),
                    'working_hours': work_hours,
                    'lat': lat,
                    'lon': lon
                })

    print(f"Extracted {len(atm_locations)} ATM locations from API")
    return atm_locations


def save_to_csv(data: List[Dict], output_path: str):
    """
    Save data to CSV file

    Args:
        data: List of dictionaries to save
        output_path: Path to output CSV file
    """
    if not data:
        print("No data to save")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Define fieldnames
    fieldnames = ['id', 'title', 'slug', 'address', 'phone', 'working_hours', 'lat', 'lon']

    print(f"Writing {len(data)} records to {output_path}...")

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Successfully saved data to {output_path}")


def main():
    """Main function"""
    # API endpoint
    api_url = "https://www.xalqbank.az/api/az/xidmet-sebekesi?include=menu"

    # Output path
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'xb_atms.csv'
    )

    try:
        # Fetch data from API
        api_data = fetch_service_network_data(api_url)

        # Extract ATM locations from API
        atm_locations = extract_atm_locations(api_data)

        # Save to CSV
        save_to_csv(atm_locations, output_path)

        print("\nDone!")
        print(f"Total ATMs: {len(atm_locations)}")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
