#!/usr/bin/env python3
"""Verify Bank Respublika addresses are populated"""

import pandas as pd

df = pd.read_csv('data/combined_locations.csv')
br = df[df['source'] == 'Bank Respublika']

print(f"Total Bank Respublika records: {len(br)}")
print(f"Records with addresses: {len(br[br['address'].notna() & (br['address'] != '')])}")
print(f"Records without addresses: {len(br[br['address'].isna() | (br['address'] == '')])}")

# Check the specific 3 records that had null addresses
print("\nChecking previously null address records:")
for loc_id in ['826', '816', '833']:
    record = br[br['location_id'] == loc_id]
    if len(record) > 0:
        name = record.iloc[0]['name']
        address = record.iloc[0]['address']
        print(f"  ID {loc_id} ({name}): {address[:50] if pd.notna(address) else 'NULL'}...")
