#!/usr/bin/env python3
"""
Strategic ATM Placement Analysis for Bank of Baku (BOB)

This script analyzes the combined location data to provide insights
for Bank of Baku to determine optimal locations for new ATMs.

Focus areas:
- Competitor ATM density analysis
- Geographic coverage gaps
- Retail/supermarket proximity opportunities
- Underserved areas identification
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, Counter
import numpy as np
from typing import List, Dict, Tuple, Optional
import math


# Set style for professional-looking charts
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def read_combined_data() -> List[Dict[str, str]]:
    """Read the combined locations dataset."""
    with open('data/combined_locations.csv', 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def safe_float(value: str) -> Optional[float]:
    """
    Safely convert a string to float, returning None if conversion fails.
    """
    try:
        # Remove any trailing commas or whitespace
        cleaned = value.strip().rstrip(',')
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


def is_valid_coords(lat: str, lon: str) -> bool:
    """Check if latitude and longitude are valid."""
    lat_f = safe_float(lat)
    lon_f = safe_float(lon)
    if lat_f is None or lon_f is None:
        return False
    # Basic range check for Azerbaijan region
    return 38 <= lat_f <= 42 and 44 <= lon_f <= 51


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on earth (in kilometers).
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def chart1_bank_atm_comparison(data: List[Dict]) -> None:
    """
    Chart 1: Bank ATM Count Comparison
    Shows how Bank of Baku compares to competitors in ATM coverage.
    """
    # Filter only bank ATMs (not supermarkets/branches)
    # Include all ATM type variations: ATM, A, atm, network_atm
    atm_types = ['ATM', 'A', 'atm', 'network_atm']
    bank_atms = [loc for loc in data if loc['type'] in atm_types]

    # Count by source
    bank_counts = Counter(loc['source'] for loc in bank_atms)

    # Sort by count
    banks = sorted(bank_counts.items(), key=lambda x: x[1], reverse=True)
    names = [b[0] for b in banks]
    counts = [b[1] for b in banks]

    # Highlight Bank of Baku
    colors = ['#FF6B6B' if name == 'Bank of Baku' else '#4ECDC4' for name in names]

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(names, counts, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width + 20, bar.get_y() + bar.get_height()/2,
                f'{count}', ha='left', va='center', fontweight='bold', fontsize=11)

    ax.set_xlabel('Number of ATMs', fontsize=12, fontweight='bold')
    ax.set_title('Bank ATM Coverage Comparison in Azerbaijan\nBank of Baku vs Competitors',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('charts/1_bank_atm_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Created: charts/1_bank_atm_comparison.png")
    plt.close()


def chart2_geographic_distribution(data: List[Dict]) -> None:
    """
    Chart 2: Geographic Distribution - BOB vs All Competitors
    Shows where BOB ATMs are located vs competitors on a map.
    """
    # Separate BOB from competitors (ATMs only)
    atm_types = ['ATM', 'A', 'atm', 'network_atm']
    bob_atms = [loc for loc in data if loc['source'] == 'Bank of Baku' and loc['type'] in atm_types]
    competitor_atms = [loc for loc in data if loc['source'] != 'Bank of Baku'
                       and loc['type'] in atm_types]

    # Extract coordinates with validation
    bob_coords = []
    for loc in bob_atms:
        if is_valid_coords(loc.get('latitude', ''), loc.get('longitude', '')):
            bob_coords.append((safe_float(loc['latitude']), safe_float(loc['longitude'])))

    comp_coords = []
    for loc in competitor_atms:
        if is_valid_coords(loc.get('latitude', ''), loc.get('longitude', '')):
            comp_coords.append((safe_float(loc['latitude']), safe_float(loc['longitude'])))

    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot competitor ATMs
    if comp_coords:
        comp_lats, comp_lons = zip(*comp_coords)
        ax.scatter(comp_lons, comp_lats, c='#4ECDC4', s=20, alpha=0.4,
                  label=f'Competitor ATMs ({len(comp_coords)})', edgecolors='none')

    # Plot BOB ATMs
    if bob_coords:
        bob_lats, bob_lons = zip(*bob_coords)
        ax.scatter(bob_lons, bob_lats, c='#FF6B6B', s=150, alpha=0.9,
                  label=f'Bank of Baku ATMs ({len(bob_coords)})',
                  edgecolors='darkred', linewidths=2, marker='D')

    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title('Geographic Distribution: Bank of Baku vs Competitors\nRed Diamonds = BOB ATMs | Teal Dots = Competitor ATMs',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('charts/2_geographic_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Created: charts/2_geographic_distribution.png")
    plt.close()


def chart3_competitor_density_heatmap(data: List[Dict]) -> None:
    """
    Chart 3: Competitor ATM Density Heatmap
    Identifies high-density areas where competitors cluster.
    """
    # Get all competitor ATMs (excluding BOB) with valid coords
    atm_types = ['ATM', 'A', 'atm', 'network_atm']
    competitor_atms = [loc for loc in data if loc['source'] != 'Bank of Baku'
                       and loc['type'] in atm_types
                       and is_valid_coords(loc.get('latitude', ''), loc.get('longitude', ''))]

    if not competitor_atms:
        return

    lats = [safe_float(loc['latitude']) for loc in competitor_atms]
    lons = [safe_float(loc['longitude']) for loc in competitor_atms]

    fig, ax = plt.subplots(figsize=(16, 12))

    # Create 2D histogram (heatmap)
    heatmap, xedges, yedges = np.histogram2d(lons, lats, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='YlOrRd',
                   aspect='auto', alpha=0.7, interpolation='gaussian')

    # Add BOB ATMs on top
    bob_atms = [loc for loc in data if loc['source'] == 'Bank of Baku'
                and is_valid_coords(loc.get('latitude', ''), loc.get('longitude', ''))]
    if bob_atms:
        bob_lats = [safe_float(loc['latitude']) for loc in bob_atms]
        bob_lons = [safe_float(loc['longitude']) for loc in bob_atms]
        ax.scatter(bob_lons, bob_lats, c='blue', s=200, alpha=1,
                  label='Bank of Baku ATMs', edgecolors='white', linewidths=2, marker='*')

    plt.colorbar(im, ax=ax, label='Competitor ATM Density')
    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title('Competitor ATM Density Heatmap\nRed = High Density | Yellow = Low Density | Blue Stars = BOB ATMs',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig('charts/3_competitor_density_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Created: charts/3_competitor_density_heatmap.png")
    plt.close()


def chart4_retail_opportunity_map(data: List[Dict]) -> None:
    """
    Chart 4: Retail & Supermarket Opportunity Map
    Shows supermarket locations (high foot traffic) and their proximity to BOB ATMs.
    """
    # Get supermarkets/retail stores (Bazarstore, Bravo, OBA branches)
    retail = [loc for loc in data if loc['source'] in ['Bazarstore', 'Bravo Supermarket', 'OBA Bank']
              and is_valid_coords(loc.get('latitude', ''), loc.get('longitude', ''))]

    bob_atms = [loc for loc in data if loc['source'] == 'Bank of Baku'
                and is_valid_coords(loc.get('latitude', ''), loc.get('longitude', ''))]

    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot retail locations by type
    colors_map = {
        'Bazarstore': '#FF9F1C',
        'Bravo Supermarket': '#2EC4B6',
        'OBA Bank': '#E71D36'
    }

    # Label mapping to clarify these are SUPERMARKETS, not banks
    label_map = {
        'Bazarstore': 'Bazarstore (Grocery Chain)',
        'Bravo Supermarket': 'Bravo Supermarket',
        'OBA Bank': 'OBA Supermarkets (NOT a bank)'
    }

    for source, color in colors_map.items():
        locs = [loc for loc in retail if loc['source'] == source]
        if locs:
            lats = [safe_float(loc['latitude']) for loc in locs]
            lons = [safe_float(loc['longitude']) for loc in locs]
            ax.scatter(lons, lats, c=color, s=50, alpha=0.6,
                      label=f'{label_map[source]} ({len(locs)})', edgecolors='black', linewidths=0.5)

    # Plot BOB ATMs
    if bob_atms:
        bob_lats = [safe_float(loc['latitude']) for loc in bob_atms]
        bob_lons = [safe_float(loc['longitude']) for loc in bob_atms]
        ax.scatter(bob_lons, bob_lats, c='blue', s=250, alpha=1,
                  label=f'Bank of Baku ATMs ({len(bob_atms)})',
                  edgecolors='white', linewidths=3, marker='*', zorder=5)

    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title('SUPERMARKET & RETAIL CHAIN Locations: High Foot Traffic Opportunities\nBlue Stars = BOB ATMs | Colored Dots = SUPERMARKET/RETAIL STORES (NOT Banks)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('charts/4_retail_opportunity_map.png', dpi=300, bbox_inches='tight')
    print("✓ Created: charts/4_retail_opportunity_map.png")
    plt.close()


def chart5_coverage_gap_analysis(data: List[Dict]) -> None:
    """
    Chart 5: Coverage Gap Analysis
    Identifies areas with competitor ATMs but no BOB presence.
    """
    atm_types = ['ATM', 'A', 'atm', 'network_atm']
    bob_atms = [(safe_float(loc['latitude']), safe_float(loc['longitude']))
                for loc in data if loc['source'] == 'Bank of Baku'
                and is_valid_coords(loc.get('latitude', ''), loc.get('longitude', ''))]

    competitor_atms = [loc for loc in data if loc['source'] != 'Bank of Baku'
                       and loc['type'] in atm_types
                       and is_valid_coords(loc.get('latitude', ''), loc.get('longitude', ''))]

    # Find competitor ATMs that are far from any BOB ATM
    gaps = []
    threshold_km = 2.0  # 2km threshold

    for comp in competitor_atms:
        comp_lat = safe_float(comp['latitude'])
        comp_lon = safe_float(comp['longitude'])

        # Find distance to nearest BOB ATM
        if bob_atms:
            min_distance = min(haversine_distance(comp_lat, comp_lon, bob_lat, bob_lon)
                              for bob_lat, bob_lon in bob_atms)
        else:
            min_distance = float('inf')

        if min_distance > threshold_km:
            gaps.append({
                'lat': comp_lat,
                'lon': comp_lon,
                'source': comp['source'],
                'distance_to_bob': min_distance
            })

    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot all competitor ATMs
    comp_lats = [safe_float(loc['latitude']) for loc in competitor_atms]
    comp_lons = [safe_float(loc['longitude']) for loc in competitor_atms]
    ax.scatter(comp_lons, comp_lats, c='lightgray', s=20, alpha=0.3,
              label=f'All Competitor ATMs ({len(competitor_atms)})')

    # Plot gaps (opportunity areas)
    if gaps:
        gap_lats = [g['lat'] for g in gaps]
        gap_lons = [g['lon'] for g in gaps]
        distances = [g['distance_to_bob'] for g in gaps]

        scatter = ax.scatter(gap_lons, gap_lats, c=distances, s=80, alpha=0.8,
                           cmap='Reds', edgecolors='darkred', linewidths=1,
                           label=f'Coverage Gaps ({len(gaps)})', vmin=threshold_km, vmax=max(distances))
        plt.colorbar(scatter, ax=ax, label='Distance to Nearest BOB ATM (km)')

    # Plot BOB ATMs
    if bob_atms:
        bob_lats, bob_lons = zip(*bob_atms)
        ax.scatter(bob_lons, bob_lats, c='blue', s=250, alpha=1,
                  label=f'Bank of Baku ATMs ({len(bob_atms)})',
                  edgecolors='white', linewidths=3, marker='*', zorder=5)

    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title(f'Coverage Gap Analysis: Areas >{threshold_km}km from BOB ATMs\nRed = Competitor ATMs with No BOB Presence',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('charts/5_coverage_gap_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Created: charts/5_coverage_gap_analysis.png")
    plt.close()

    return gaps


def chart6_market_share_by_region(data: List[Dict]) -> None:
    """
    Chart 6: Market Share Analysis by Geographic Quadrant
    Divides the map into regions and shows market share.
    """
    # Get all bank ATMs with coordinates
    atm_types = ['ATM', 'A', 'atm', 'network_atm']
    bank_atms = [loc for loc in data if loc['type'] in atm_types
                 and is_valid_coords(loc.get('latitude', ''), loc.get('longitude', ''))]

    if not bank_atms:
        return

    # Calculate median lat/lon to divide into quadrants
    lats = [safe_float(loc['latitude']) for loc in bank_atms]
    lons = [safe_float(loc['longitude']) for loc in bank_atms]
    median_lat = np.median(lats)
    median_lon = np.median(lons)

    # Categorize into quadrants
    quadrants = {
        'North-East': [],
        'North-West': [],
        'South-East': [],
        'South-West': []
    }

    for loc in bank_atms:
        lat = safe_float(loc['latitude'])
        lon = safe_float(loc['longitude'])

        if lat >= median_lat and lon >= median_lon:
            quadrants['North-East'].append(loc)
        elif lat >= median_lat and lon < median_lon:
            quadrants['North-West'].append(loc)
        elif lat < median_lat and lon >= median_lon:
            quadrants['South-East'].append(loc)
        else:
            quadrants['South-West'].append(loc)

    # Count BOB vs competitors in each quadrant
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Regional Market Share Analysis: Bank of Baku vs Competitors',
                 fontsize=16, fontweight='bold', y=0.995)

    quadrant_names = ['North-West', 'North-East', 'South-West', 'South-East']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for quad_name, (row, col) in zip(quadrant_names, positions):
        ax = axes[row, col]
        quad_data = quadrants[quad_name]

        # Count by bank
        bank_counts = Counter(loc['source'] for loc in quad_data)
        bob_count = bank_counts.get('Bank of Baku', 0)
        total = len(quad_data)

        # Create data for chart
        banks = sorted(bank_counts.items(), key=lambda x: x[1], reverse=True)[:6]  # Top 6
        names = [b[0][:15] for b in banks]  # Truncate long names
        counts = [b[1] for b in banks]

        colors = ['#FF6B6B' if 'Baku' in name else '#4ECDC4' for name in names]

        bars = ax.barh(names, counts, color=colors, edgecolor='black', linewidth=1)

        # Add value labels
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{count}', ha='left', va='center', fontweight='bold', fontsize=9)

        ax.set_title(f'{quad_name} Region\nTotal ATMs: {total} | BOB: {bob_count} ({bob_count/total*100:.1f}%)',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('ATM Count', fontsize=9)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('charts/6_market_share_by_region.png', dpi=300, bbox_inches='tight')
    print("✓ Created: charts/6_market_share_by_region.png")
    plt.close()


def chart7_nearest_retail_opportunities(data: List[Dict]) -> None:
    """
    Chart 7: Top Retail Locations Without Nearby BOB ATMs
    Identifies specific retail locations that would be ideal for new BOB ATMs.
    """
    # Get retail locations
    retail = [loc for loc in data if loc['source'] in ['Bazarstore', 'Bravo Supermarket', 'OBA Bank']
              and is_valid_coords(loc.get('latitude', ''), loc.get('longitude', ''))]

    # Get BOB ATMs
    bob_atms = [(safe_float(loc['latitude']), safe_float(loc['longitude']))
                for loc in data if loc['source'] == 'Bank of Baku'
                and is_valid_coords(loc.get('latitude', ''), loc.get('longitude', ''))]

    # Calculate distance from each retail to nearest BOB ATM
    opportunities = []

    for loc in retail:
        lat = safe_float(loc['latitude'])
        lon = safe_float(loc['longitude'])

        if bob_atms:
            min_dist = min(haversine_distance(lat, lon, bob_lat, bob_lon)
                          for bob_lat, bob_lon in bob_atms)
        else:
            min_dist = float('inf')

        opportunities.append({
            'name': loc['name'][:40],  # Truncate name
            'source': loc['source'],
            'distance': min_dist,
            'address': loc['address'][:50]
        })

    # Sort by distance and get top 20
    opportunities.sort(key=lambda x: x['distance'], reverse=True)
    top_opportunities = opportunities[:20]

    fig, ax = plt.subplots(figsize=(14, 10))

    # Create clear labels showing these are SUPERMARKETS, not banks
    label_map = {
        'OBA Bank': 'OBA Supermarket',
        'Bravo Supermarket': 'Bravo Supermarket',
        'Bazarstore': 'Bazarstore Grocery'
    }

    names = [f"{o['name']}\n({label_map.get(o['source'], o['source'])})" for o in top_opportunities]
    distances = [o['distance'] for o in top_opportunities]

    colors = ['#E71D36' if 'OBA' in o['source'] else
              '#FF9F1C' if 'Bazarstore' in o['source'] else '#2EC4B6'
              for o in top_opportunities]

    bars = ax.barh(range(len(names)), distances, color=colors, edgecolor='black', linewidth=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)

    # Add distance labels
    for i, (bar, dist) in enumerate(zip(bars, distances)):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
               f'{dist:.1f} km', ha='left', va='center', fontweight='bold', fontsize=9)

    ax.set_xlabel('Distance to Nearest BOB ATM (km)', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 SUPERMARKET/RETAIL Partnership Opportunities for New BOB ATMs\nThese are RETAIL STORES (NOT Banks) - Furthest from Current BOB Coverage',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)

    # Add legend with clear supermarket labels
    patches = [
        mpatches.Patch(color='#E71D36', label='OBA Supermarkets (NOT a bank)'),
        mpatches.Patch(color='#FF9F1C', label='Bazarstore (Grocery Chain)'),
        mpatches.Patch(color='#2EC4B6', label='Bravo Supermarket')
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig('charts/7_nearest_retail_opportunities.png', dpi=300, bbox_inches='tight')
    print("✓ Created: charts/7_nearest_retail_opportunities.png")
    plt.close()

    return top_opportunities


def generate_insights_report(data: List[Dict], gaps: List[Dict], retail_opps: List[Dict]) -> str:
    """
    Generate a comprehensive insights report for Bank of Baku.
    """
    atm_types = ['ATM', 'A', 'atm', 'network_atm']
    bob_atms = [loc for loc in data if loc['source'] == 'Bank of Baku']
    all_bank_atms = [loc for loc in data if loc['type'] in atm_types]
    retail_locations = [loc for loc in data if loc['source'] in ['Bazarstore', 'Bravo Supermarket', 'OBA Bank']]

    bob_count = len(bob_atms)
    total_bank_atms = len(all_bank_atms)
    market_share = (bob_count / total_bank_atms * 100) if total_bank_atms else 0

    competitor_counts = Counter(loc['source'] for loc in all_bank_atms if loc['source'] != 'Bank of Baku')
    top_competitor = competitor_counts.most_common(1)[0] if competitor_counts else ('N/A', 0)

    report = f"""
STRATEGIC ATM PLACEMENT INSIGHTS FOR BANK OF BAKU
================================================

CURRENT MARKET POSITION
-----------------------
• Total BOB ATMs: {bob_count}
• Total Market ATMs: {total_bank_atms}
• BOB Market Share: {market_share:.1f}%
• Top Competitor: {top_competitor[0]} with {top_competitor[1]} ATMs
• Gap to Leader: {top_competitor[1] - bob_count} ATMs

COMPETITIVE LANDSCAPE
---------------------
Top 5 Competitors by ATM Count:
"""

    for i, (bank, count) in enumerate(competitor_counts.most_common(5), 1):
        report += f"  {i}. {bank}: {count} ATMs\n"

    report += f"""
COVERAGE GAP ANALYSIS
---------------------
• Identified {len(gaps) if gaps else 0} high-priority coverage gaps
• These are areas with competitor ATMs but no BOB presence within 2km
• Recommendation: Prioritize these areas for immediate expansion

RETAIL PARTNERSHIP OPPORTUNITIES
---------------------------------
• Total retail locations analyzed: {len(retail_locations)}
  - OBA Bank branches: {len([r for r in retail_locations if r['source'] == 'OBA Bank'])}
  - Bazarstore locations: {len([r for r in retail_locations if r['source'] == 'Bazarstore'])}
  - Bravo Supermarkets: {len([r for r in retail_locations if r['source'] == 'Bravo Supermarket'])}

TOP STRATEGIC RECOMMENDATIONS
------------------------------
1. IMMEDIATE PRIORITY: Fill coverage gaps
   - {len(gaps) if gaps else 0} locations identified where competitors operate but BOB doesn't
   - Focus on areas >2km from current BOB ATMs

2. RETAIL PARTNERSHIPS: High foot-traffic locations
   - Partner with major retail chains (Bazarstore, Bravo, OBA branches)
   - Top 20 retail opportunities identified with no nearby BOB presence

3. GEOGRAPHIC EXPANSION: Balance regional presence
   - Analyze regional market share charts to identify underserved quadrants
   - Consider both urban density and regional coverage

4. COMPETITIVE STRATEGY:
   - Current deficit vs leader: {top_competitor[1] - bob_count} ATMs
   - Recommended target: Add 50-100 ATMs in strategic locations
   - Focus on high-density areas and retail partnerships

5. DATA-DRIVEN PLACEMENT:
   - Use heatmaps to avoid over-saturation
   - Prioritize areas with competitor presence (proven demand)
   - Balance between competing with leaders and serving underserved areas

NEXT STEPS
----------
1. Review all generated charts in /charts folder
2. Cross-reference coverage gaps with retail opportunities
3. Conduct on-ground feasibility studies for top 20 retail locations
4. Develop partnership agreements with high-traffic retail chains
5. Create phased rollout plan (Phase 1: 20 ATMs, Phase 2: 30 ATMs, etc.)

METHODOLOGY
-----------
Analysis based on {len(data)} total locations including:
- {len(all_bank_atms)} bank ATMs from {len(set(loc['source'] for loc in all_bank_atms))} banks
- {len(retail_locations)} retail/branch locations
- Geographic clustering and distance calculations using Haversine formula
- 2km radius used as standard ATM service area

Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    return report


def main():
    """Main execution function."""
    print("Starting strategic analysis for Bank of Baku ATM placement...\n")

    # Read data
    data = read_combined_data()
    print(f"Loaded {len(data)} total locations\n")

    # Generate charts
    print("Generating visualizations...")
    chart1_bank_atm_comparison(data)
    chart2_geographic_distribution(data)
    chart3_competitor_density_heatmap(data)
    chart4_retail_opportunity_map(data)
    gaps = chart5_coverage_gap_analysis(data)
    chart6_market_share_by_region(data)
    retail_opps = chart7_nearest_retail_opportunities(data)

    # Generate insights report
    print("\nGenerating insights report...")
    report = generate_insights_report(data, gaps or [], retail_opps or [])

    with open('charts/INSIGHTS_REPORT.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print("✓ Created: charts/INSIGHTS_REPORT.txt")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nGenerated 7 charts in /charts folder")
    print("Review INSIGHTS_REPORT.txt for detailed strategic recommendations")
    print("\nCharts created:")
    print("  1. Bank ATM comparison")
    print("  2. Geographic distribution map")
    print("  3. Competitor density heatmap")
    print("  4. Retail opportunity map")
    print("  5. Coverage gap analysis")
    print("  6. Market share by region")
    print("  7. Top retail opportunities")


if __name__ == "__main__":
    main()
