#!/usr/bin/env python3
"""
Script to fetch Bank of Baku ATM locations and save to CSV
"""

import requests
import csv
import json
from typing import List, Dict, Optional
import os


# Predefined list of ATM names to search for
ATM_NAMES = [
    "Baş İdarə ATM",
    "«Azneft» filialının ATM-i",
    "«Bakıxanov» filialının ATM-i",
    "Əməliyyat mərkəzidə ATM",
    "«Əhmədli» filialının ATM-i",
    "«Həzi Aslanov» filialının ATM-i",
    "«Mərkəz» filialının ATM-i",
    "«Mərdəkan» filialının ATM-i",
    "«Nəsimi» filialındakı ATM",
    "«Neftçilər» filialının ATM-i",
    "«Otoplaza» filialının ATM-i",
    "«Səməd Vurğun» filialının ATM-i",
    "«Yasamal» filialının ATM-i",
    "«Nərimanov» filialının ATM-i",
    "«Xırdalan» filialının ATM-i",
    "«Gəncə» filialının ATM-i",
    "«Yeni Gəncə» filialının ATM-i",
    "«Lənkəran» filialının ATM-i",
    "«Sumqayıt» filialının ATM-i",
    "«Şəki» filialının ATM-i",
    "«Şirvan» filialının ATM-i",
    "«Xaçmaz» filialının ATM-i",
    "«Park Bulvar»-da ATM",
    "28 Mall T/M",
    "Favorit Market (Mir Cəlal 59)",
    "Baku Electronics-də ATM Ə.Naxçıvani",
    "Improtex travel",
    "Demirchi Tower (ATM)",
    "Memarliq Universitet",
    "Nizami Mall",
    "Qəbələ Futbol Akademiyası",
    "EmbaFinans",
    "Inşaatçılar Filialı",
    "Baku Tobacco",
    "Metro Park TM-ATM",
    "Gənclik Mall TM"
]


def fetch_service_network_data(url: str) -> Dict:
    """
    Fetch service network data from Bank of Baku API

    Args:
        url: API endpoint URL

    Returns:
        API response dictionary
    """
    print(f"Fetching data from {url}...")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Origin': 'https://www.bankofbaku.com',
        'Referer': 'https://www.bankofbaku.com/'
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

    payload = data.get('payload', {})
    information_groups = payload.get('informationGroup', [])

    for group in information_groups:
        lists = group.get('lists', [])

        for item in lists:
            # Filter for Azerbaijani language entries
            if item.get('language') == 'az':
                # Parse location coordinates
                location_str = item.get('location', '')
                lat, lon = None, None
                if location_str:
                    try:
                        parts = location_str.split(',')
                        if len(parts) == 2:
                            lat = float(parts[0].strip())
                            lon = float(parts[1].strip())
                    except (ValueError, AttributeError):
                        pass

                atm_locations.append({
                    'title': item.get('title'),
                    'address': item.get('address'),
                    'service_names': item.get('serviceNames'),
                    'lat': lat,
                    'lon': lon,
                    'position_order': group.get('positionOrder')
                })

    print(f"Extracted {len(atm_locations)} locations from API")
    return atm_locations


def normalize_string(s: str) -> str:
    """
    Normalize string for fuzzy matching by removing special characters

    Args:
        s: Input string

    Returns:
        Normalized string
    """
    if not s:
        return ""

    # Remove quotes, convert to lowercase
    s = s.replace('«', '').replace('»', '').replace('"', '').replace("'", '')
    s = s.lower().strip()
    return s


def match_atms(atm_names: List[str], api_locations: List[Dict]) -> List[Dict]:
    """
    Match predefined ATM names with API locations

    Args:
        atm_names: List of ATM names to search for
        api_locations: List of locations from API

    Returns:
        List of matched ATM dictionaries
    """
    matched_atms = []

    print("\nMatching ATMs...")

    for atm_name in atm_names:
        normalized_name = normalize_string(atm_name)

        # Try to find a match in API locations
        matched = False
        for location in api_locations:
            location_title = normalize_string(location.get('title', ''))

            # Check if the normalized titles match (fuzzy)
            if location_title and normalized_name in location_title or location_title in normalized_name:
                matched_atms.append({
                    'name': atm_name,
                    'api_title': location.get('title'),
                    'address': location.get('address'),
                    'service_names': location.get('service_names'),
                    'lat': location.get('lat'),
                    'lon': location.get('lon'),
                    'position_order': location.get('position_order'),
                    'matched': True
                })
                matched = True
                print(f"✓ Matched: {atm_name}")
                break

        if not matched:
            # Add ATM without coordinates if not found
            matched_atms.append({
                'name': atm_name,
                'api_title': None,
                'address': None,
                'service_names': None,
                'lat': None,
                'lon': None,
                'position_order': None,
                'matched': False
            })
            print(f"✗ Not matched: {atm_name}")

    matched_count = sum(1 for atm in matched_atms if atm['matched'])
    print(f"\nMatched {matched_count}/{len(atm_names)} ATMs")

    return matched_atms


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
    fieldnames = ['name', 'api_title', 'address', 'service_names', 'lat', 'lon', 'position_order', 'matched']

    print(f"\nWriting {len(data)} records to {output_path}...")

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Successfully saved data to {output_path}")


def main():
    """Main function"""
    # API endpoint
    api_url = "https://site-api.bankofbaku.com/categories/serviceNetwork/individual"

    # Output path
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'bob_atms.csv'
    )

    try:
        # Fetch data from API
        api_data = fetch_service_network_data(api_url)

        # Extract ATM locations from API
        api_locations = extract_atm_locations(api_data)

        # Match predefined ATM names with API data
        matched_atms = match_atms(ATM_NAMES, api_locations)

        # Save to CSV
        save_to_csv(matched_atms, output_path)

        print("\nDone!")
        print(f"Total ATMs: {len(matched_atms)}")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
