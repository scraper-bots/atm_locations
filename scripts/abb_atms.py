#!/usr/bin/env python3
"""
Script to fetch ABB Bank ATM locations and save to CSV
"""

import requests
import csv
import json
from typing import List, Dict
import os


def fetch_atm_data(url: str) -> List[Dict]:
    """
    Fetch ATM location data from ABB Bank API

    Args:
        url: API endpoint URL

    Returns:
        List of ATM location dictionaries
    """
    print(f"Fetching data from {url}...")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'az'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    response_data = response.json()
    # Extract the 'data' array from the response
    data = response_data.get('data', [])
    print(f"Successfully fetched {len(data)} locations")

    return data


def flatten_location_data(locations: List[Dict]) -> List[Dict]:
    """
    Flatten the location data for CSV export

    Args:
        locations: List of location dictionaries

    Returns:
        Flattened list of dictionaries
    """
    flattened = []

    for location in locations:
        # Extract location coordinates
        location_obj = location.get('location', {})
        lat = location_obj.get('lat') if location_obj else None
        lon = location_obj.get('lon') if location_obj else None

        # Create flattened record
        record = {
            'contentful_id': location.get('contentfulId'),
            'ext_id': location.get('extId'),
            'name': location.get('name'),
            'type': location.get('type'),
            'address': location.get('address'),
            'lat': lat,
            'lon': lon,
            'status': location.get('status'),
            'atm_cash_in': location.get('atmCashIn'),
            'work_on_weekend': location.get('workOnWeekend'),
            'full_day': location.get('fullDay'),
            'working_time': location.get('workingTime'),
            'weekdays': location.get('weekdays'),
            'saturdays': location.get('saturdays'),
            'phone': location.get('phone'),
            'atm_branch_image': location.get('atmbranchImage')
        }

        flattened.append(record)

    return flattened


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

    # Get all fieldnames from data
    fieldnames = list(data[0].keys())

    print(f"Writing {len(data)} records to {output_path}...")

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Successfully saved data to {output_path}")


def main():
    """Main function"""
    # API endpoint
    api_url = "https://abb-bank.az/web-api/service-network/atm"

    # Output path
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'abb_atms.csv'
    )

    try:
        # Fetch data
        locations = fetch_atm_data(api_url)

        # Flatten data
        flattened_data = flatten_location_data(locations)

        # Save to CSV
        save_to_csv(flattened_data, output_path)

        print("\nDone!")
        print(f"Total locations: {len(flattened_data)}")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
