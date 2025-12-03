#!/usr/bin/env python3
"""
Bazarstore Branch Location Scraper

This script fetches branch location data from Bazarstore's API via Stockist.
The data is provided as a JSON array with detailed location information.

Data extracted:
- Branch name
- Address (street, city, country)
- Latitude and longitude coordinates
- Contact information (phone, email, website)
- Additional metadata (image URL, description)
"""

import csv
import json
import requests
from typing import List, Dict, Optional


def fetch_bazarstore_locations() -> Optional[List[Dict]]:
    """
    Fetch branch locations from Bazarstore's Stockist API.

    Returns:
        List of location dictionaries, or None if request fails
    """
    url = "https://stockist.co/api/v1/u17579/locations/all"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
        "Referer": "https://bazarstore.az/",
        "Accept": "application/json",
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Bazarstore locations: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return None


def parse_location_data(locations: List[Dict]) -> List[Dict[str, str]]:
    """
    Convert raw API location data into structured branch dictionaries.

    Args:
        locations: List of location dictionaries from API

    Returns:
        List of dictionaries with structured branch information
    """
    branches = []

    for location in locations:
        # Combine address lines if address_line_2 exists
        address = location.get("address_line_1", "")
        if location.get("address_line_2"):
            address = f"{address}, {location['address_line_2']}"

        branch = {
            "id": str(location.get("id", "")),
            "name": location.get("name", ""),
            "address": address,
            "city": location.get("city", ""),
            "country": location.get("country", ""),
            "postal_code": location.get("postal_code", ""),
            "latitude": str(location.get("latitude", "")),
            "longitude": str(location.get("longitude", "")),
            "phone": location.get("phone", ""),
            "email": location.get("email", ""),
            "website": location.get("website", ""),
            "description": location.get("description", ""),
            "image_url": location.get("image_url", "")
        }
        branches.append(branch)

    return branches


def save_to_csv(branches: List[Dict[str, str]], filename: str = "data/bazarstore_branches.csv") -> None:
    """
    Save branch data to CSV file.

    Args:
        branches: List of branch dictionaries
        filename: Output CSV filename
    """
    if not branches:
        print("No branch data to save")
        return

    fieldnames = [
        "id", "name", "address", "city", "country", "postal_code",
        "latitude", "longitude", "phone", "email", "website",
        "description", "image_url"
    ]

    try:
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(branches)

        print(f"Successfully saved {len(branches)} branches to {filename}")
    except IOError as e:
        print(f"Error writing to CSV: {e}")


def main():
    """Main execution function."""
    print("Fetching Bazarstore branch data...")

    # Fetch locations from API
    locations = fetch_bazarstore_locations()
    if not locations:
        print("Failed to fetch location data")
        return

    print(f"Found {len(locations)} branches")

    # Parse into structured data
    branches = parse_location_data(locations)

    # Save to CSV
    save_to_csv(branches)


if __name__ == "__main__":
    main()
