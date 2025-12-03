#!/usr/bin/env python3
"""
Bank Respublika ATM and Branch Location Scraper

This script scrapes ATM and branch location data from Bank Respublika's website
and saves it to a CSV file.
"""

import requests
import csv
import json
import re
from typing import List, Dict
from html import unescape


def fetch_page_data(url: str) -> str:
    """
    Fetch the HTML page containing ATM and branch data.

    Args:
        url: The URL to fetch

    Returns:
        HTML content as string
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text


def clean_html(text: str) -> str:
    """
    Remove HTML tags and clean up text.

    Args:
        text: Text potentially containing HTML tags

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode HTML entities
    text = unescape(text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_location_data(html: str) -> List[Dict]:
    """
    Extract ATM and branch location data from HTML.

    Args:
        html: HTML content containing location data

    Returns:
        List of dictionaries containing location information
    """
    locations = []

    # Find all data-info attributes - they contain JSON with &quot; entities
    # We need to extract the content between data-info=" and the closing "
    # The JSON itself contains &quot; so we can't simply match until the first "

    # Split by data-info=" to get each JSON block
    parts = html.split('data-info="')

    print(f"Found {len(parts) - 1} potential locations")

    for i in range(1, len(parts)):  # Skip first part (before any data-info)
        part = parts[i]

        # Find the end of the JSON - it ends with }" followed by a space or >
        # We need to find the closing of the JSON object
        end_match = re.search(r'\}"\s*[>\s]', part)
        if not end_match:
            continue

        json_str_encoded = part[:end_match.start() + 1]

        try:
            # Decode HTML entities in JSON string
            json_str = unescape(json_str_encoded)
            # Parse JSON
            data = json.loads(json_str)

            # Extract extras field
            extras = data.get('extras', {})

            # Determine type (branch or ATM)
            categories = data.get('categorylist', [])
            is_branch = 'branches' in categories
            is_atm = 'atms' in categories

            # Skip if it's only a branch (not an ATM)
            if not is_atm:
                continue

            # Clean up the shortstory field which contains address/phone/email/hours
            shortstory = clean_html(data.get('shortstory', ''))

            location = {
                'id': data.get('id', ''),
                'title': data.get('title', ''),
                'slug': data.get('slug', ''),
                'type': 'branch' if is_branch else 'atm',
                'categories': ', '.join(categories),
                'info': shortstory,
                'latitude': extras.get('lattitude', ''),  # Note: API uses 'lattitude' (typo)
                'longitude': extras.get('longitude', ''),
                'city_location': extras.get('citylocation', ''),
                'branch_id': extras.get('branchid', ''),
                'cash_in': extras.get('cashin', ''),
                'atm_currencies': extras.get('atmscurrencies', ''),
                'date': data.get('date', '')
            }

            locations.append(location)

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            continue
        except Exception as e:
            print(f"Error processing location: {e}")
            continue

    return locations


def save_to_csv(locations: List[Dict], filename: str) -> None:
    """
    Save location data to CSV file.

    Args:
        locations: List of location dictionaries
        filename: Output CSV filename
    """
    if not locations:
        print("No locations to save")
        return

    # Define CSV columns
    fieldnames = [
        'id',
        'title',
        'slug',
        'type',
        'categories',
        'info',
        'latitude',
        'longitude',
        'city_location',
        'branch_id',
        'cash_in',
        'atm_currencies',
        'date'
    ]

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(locations)

    print(f"Saved {len(locations)} locations to {filename}")


def main():
    """Main function to orchestrate the scraping process."""
    print("Starting Bank Respublika ATM and Branch scraper...")

    # Bank Respublika branches and ATMs page
    url = "https://www.bankrespublika.az/en/atms"

    try:
        # Fetch page data
        print(f"Fetching data from {url}...")
        html = fetch_page_data(url)

        # Extract location data
        print("Extracting location data...")
        locations = extract_location_data(html)

        # Save to CSV
        output_file = "../data/br_atms.csv"
        print(f"Saving to {output_file}...")
        save_to_csv(locations, output_file)

        # Print summary statistics
        print("\n=== Summary ===")
        print(f"Total ATMs: {len(locations)}")

        # Count by city location
        baku_count = len([loc for loc in locations if loc['city_location'] == 'baku'])
        regions_count = len([loc for loc in locations if loc['city_location'] == 'regions'])

        print(f"Baku: {baku_count}")
        print(f"Regions: {regions_count}")

        # Count ATMs with cash-in
        cash_in_count = len([loc for loc in locations if loc['cash_in'] == '1'])
        print(f"\nATMs with cash-in capability: {cash_in_count}")

        # Count by currency support
        multi_currency = len([loc for loc in locations if loc['atm_currencies'] and 'usd' in loc['atm_currencies']])
        print(f"Multi-currency ATMs (USD/EUR): {multi_currency}")

        print("\n✓ Bank Respublika scraping completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
