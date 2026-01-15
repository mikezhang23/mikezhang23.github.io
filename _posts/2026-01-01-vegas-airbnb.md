---
layout: post
title: Las Vegas Short-Term Rental Investment Analysis - Identifying 18% ROI Opportunities
image: "/posts/vegas-airbnb.jpeg"
tags: [Real Estate Analytics, Python, SQL, API Integration, Web Scraping, Data Visualization]
---

Our analysis of 17,600+ Airbnb listings, 6.4 million booking records, and 3,000 MLS home sales revealed that 2-bedroom properties in Las Vegas's Unincorporated Areas (the Strip) deliver the highest ROI at 18% for top operators. Through comprehensive SQL analysis, revenue modeling, and hotel price scraping, we identified that properties with pools generate 93% more revenue and that Airbnb offers guests 10-30% savings versus comparable hotels. Let's dive into how data science can optimize real estate investment decisions!

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
    - [Key Definitions](#overview-definitions)
- [01. Data Overview](#data-overview)
- [02. Data Cleaning](#data-cleaning)
- [03. SQL Analysis](#sql-analysis)
- [04. Occupancy & Revenue Calculation](#occupancy-revenue)
- [05. ROI Analysis](#roi-analysis)
- [06. Amenity Impact Analysis](#amenity-analysis)
- [07. Hotel Price Scraping](#hotel-scraping)
- [08. Hotel vs Airbnb Comparison](#hotel-comparison)
- [09. Visualizations](#visualizations)
- [10. Investment Recommendations](#recommendations)
- [11. Growth & Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

As a short-term rental investor with three properties in Las Vegas, I wanted to answer a fundamental question: **What type of property should I buy next to maximize ROI?**

The challenge with real estate investment is that average market data can be misleading. Poorly managed properties drag down averages, while top operators achieve significantly higher returns. Additionally, factors like amenities, location, and property size all interact to affect both nightly rates (ADR) and occupancy—the two components that drive revenue.

The overall aim was to build a comprehensive analysis framework that combines Airbnb performance data with actual home purchase prices, identifies revenue-driving amenities, and benchmarks against hotel pricing to create actionable investment recommendations.

<br>
### Actions <a name="overview-actions"></a>

We approached this challenge through a comprehensive four-phase analysis:

**Phase 1: Data Collection & Cleaning**
* Loaded Airbnb listings, calendar, and neighbourhood data from Inside Airbnb
* Imported 6 MLS datasets covering 1-6 bedroom home sales
* Cleaned price fields, removed outliers ($10-$1,000/night range)
* Calculated occupancy rates from 6.4M calendar records

**Phase 2: SQL Analysis**
* Created SQLite database with 7 integrated tables
* Analyzed revenue by room type, neighbourhood, and bedroom count
* Calculated ROI using 75th and 90th percentile performers
* Identified top revenue-driving amenities

**Phase 3: Hotel Price Scraping (API)**
* Connected to Booking.com API via RapidAPI
* Scraped prices across 8 date ranges (4 full weeks + 4 weekends)
* Analyzed by star rating and room type
* Built competitive pricing comparison

**Phase 4: Visualization & Recommendations**
* Created comparison charts for ROI and pricing
* Developed prioritized investment recommendations
* Segmented by bedroom count, location, and amenities

<br>
### Results <a name="overview-results"></a>

Our analysis uncovered **18% ROI opportunities** for top-performing operators:

**Key Findings:**
* 2-bedroom properties deliver highest ROI (7.9% at 75th percentile, 18.0% at 90th)
* Pools increase revenue by 93%, hot tubs by 68%, saunas by 81%
* Unincorporated Areas (the Strip) generates $28,763 average annual revenue
* Airbnb offers 10-30% savings vs comparable hotel rooms

**Revenue by Operator Skill Level:**

| Bedrooms | Good Operator (75th %) | Excellent Operator (90th %) |
|----------|------------------------|----------------------------|
| 2 BR | $50,322/year | $86,625/year |
| 5 BR | $65,182/year | $107,831/year |
| 6 BR | $92,225/year | $148,000/year |

**Investment Comparison:**
* Best ROI: 2-bedroom (18.0% for excellent operators)
* Best Cash Flow: 6-bedroom ($108K net income for excellent operators)

<br>
### Growth/Next Steps <a name="overview-growth"></a>

The framework we built is immediately actionable:

**Immediate Actions:**
* Target 2-bedroom properties in Unincorporated Areas with pools
* Budget $361K for purchase, expect $65K-$87K annual revenue
* Prioritize properties with hot tubs, pool tables, or game rooms

**Medium-term Enhancements:**
* Add seasonal analysis (revenue by month)
* Include detailed expense breakdown (cleaning, management, utilities)
* Build interactive Streamlit dashboard
* Expand to other markets (Phoenix, Austin, Miami)

<br>
### Key Definitions  <a name="overview-definitions"></a>

**ADR (Average Daily Rate):** The average nightly price for a listing

**Occupancy Rate:** Percentage of available nights that are booked (calculated from calendar data where 'f' = booked, 't' = available)

**Annual Revenue:** ADR × Occupancy Rate × 365

**ROI (Return on Investment):** (Annual Revenue - Annual Mortgage) / Home Purchase Price

**75th/90th Percentile:** Performance level of good (top 25%) and excellent (top 10%) operators, used instead of averages to filter out poorly managed properties

**Price Elasticity:** Measure of demand sensitivity to price changes

___

# Data Overview  <a name="data-overview"></a>

We analyzed 12 months of marketplace data from three primary sources, integrating them into a SQLite database for comprehensive analysis.
```python
# Import required packages
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

# Load Airbnb data
listings = pd.read_csv('listings.csv.gz', compression='gzip')
calendar = pd.read_csv('calendar.csv.gz', compression='gzip')
neighbourhoods = pd.read_csv('neighbourhoods.csv')

# Load MLS home sales data
mls_1bed = pd.read_csv('mls_1bed_6mo.csv')
mls_2bed = pd.read_csv('mls_2bed_500kmax_3mo.csv')
mls_3bed = pd.read_csv('mls_3bed_500kmax_3mo.csv')
mls_4bed = pd.read_csv('mls_4bed_500kmax_3mo.csv')
mls_5bed = pd.read_csv('mls_5bed_667kmax_3mo.csv')
mls_6bed = pd.read_csv('mls_6bed_868kmax_3mo.csv')

# Combine MLS data
mls = pd.concat([mls_1bed, mls_2bed, mls_3bed, mls_4bed, mls_5bed, mls_6bed], ignore_index=True)

# Data volume summary
print(f"Listings: {len(listings):,}")      # 17,624
print(f"Calendar: {len(calendar):,}")      # 6,432,761
print(f"Neighbourhoods: {len(neighbourhoods)}")  # 7
print(f"MLS: {len(mls):,}")                # 2,990
```

<br>
After data preprocessing and integration, our database contained:

<br>

| **Table** | **Rows** | **Description** |
|---|---|---|
| listings | 11,833 | Cleaned Airbnb listings with revenue calculations |
| mls | 2,990 | Home sales by bedroom count |
| neighbourhoods | 7 | Geographic areas in Clark County |
| top_performers | 6 | Revenue by bedroom (75th & 90th percentile) |
| amenity_impact | 15 | Amenity revenue lift analysis |
| hotels | 160 | Scraped hotel pricing data |
| hotel_vs_airbnb | 6 | Competitive comparison summary |

<br>

| **Variable Name** | **Variable Type** | **Description** |
|---|---|---|
| price_clean | Calculated | Nightly rate converted from string ($150.00) to float |
| occupancy_rate | Calculated | Percentage of nights booked per listing |
| annual_revenue | Dependent | ADR × Occupancy × 365 |
| neighbourhood_cleansed | Independent | Geographic area (7 neighbourhoods) |
| room_type | Independent | Entire home, Private room, Hotel room, Shared room |
| bedrooms | Independent | Number of bedrooms (1-6) |
| amenities | Independent | List of property amenities |
| home_price | Independent | MLS sale price by bedroom count |

___

# Data Cleaning <a name="data-cleaning"></a>

Raw Airbnb data required significant cleaning before analysis. Price fields contained dollar signs and commas, and outliers skewed averages significantly.

### Price Cleaning
```python
# Original price format: "$150.00"
listings['price_clean'] = listings['price'].str.replace('$', '').str.replace(',', '').astype(float)

# Check for outliers
print(f"Min: ${listings['price_clean'].min()}")   # $3
print(f"Max: ${listings['price_clean'].max()}")   # $100,000
print(f"Mean: ${listings['price_clean'].mean():.0f}")    # $520
print(f"Median: ${listings['price_clean'].median():.0f}") # $148
```

The massive gap between mean ($520) and median ($148) indicated severe outlier influence. A $100,000/night listing was skewing our entire analysis.

### Outlier Removal
```python
# Keep only realistic nightly rates ($10 - $1,000)
listings_clean = listings[(listings['price_clean'] >= 10) & (listings['price_clean'] <= 1000)]

print(f"Before cleaning: {len(listings):,}")      # 17,624
print(f"After cleaning: {len(listings_clean):,}") # 11,833
print(f"Removed: {len(listings) - len(listings_clean):,} listings")  # 5,791

# Verify improvement
print(f"New Mean: ${listings_clean['price_clean'].mean():.0f}")    # $182
print(f"New Median: ${listings_clean['price_clean'].median():.0f}") # $144
```

After cleaning, mean ($182) and median ($144) were much closer, indicating a normalized distribution.

### MLS Data Cleaning
```python
# Clean MLS price (remove $ and convert to number)
mls['price_clean'] = mls['Current Price'].str.replace('$', '').str.replace(',', '').astype(float)

# Clean square footage
mls['sqft'] = mls['Approx Liv Area'].str.replace(',', '').astype(float)

print(f"MLS price range: ${mls['price_clean'].min():,.0f} to ${mls['price_clean'].max():,.0f}")
# Result: $61,750 to $850,000
```

___

# SQL Analysis <a name="sql-analysis"></a>

We created a SQLite database to enable complex queries combining Airbnb performance with home purchase prices.

### Database Creation
```python
# Create SQLite database (saved to file for persistence)
conn = sqlite3.connect('vegas_airbnb.db')

# Store dataframes as SQL tables
listings_full.to_sql('listings', conn, index=False, if_exists='replace')
mls.to_sql('mls', conn, index=False, if_exists='replace')
neighbourhoods.to_sql('neighbourhoods', conn, index=False, if_exists='replace')

print("Database saved: vegas_airbnb.db")
```

### Revenue by Room Type
```sql
SELECT 
    room_type, 
    COUNT(*) as count, 
    ROUND(AVG(annual_revenue), 0) as avg_revenue
FROM listings
GROUP BY room_type
ORDER BY avg_revenue DESC
```

| room_type | count | avg_revenue |
|-----------|-------|-------------|
| Entire home/apt | 8,699 | $28,235 |
| Hotel room | 151 | $24,325 |
| Private room | 2,903 | $19,206 |
| Shared room | 80 | $7,219 |

### Revenue by Neighbourhood
```sql
SELECT 
    neighbourhood_cleansed as neighbourhood,
    COUNT(*) as listings,
    ROUND(AVG(price_clean), 0) as avg_adr,
    ROUND(AVG(occupancy_rate), 2) as avg_occupancy,
    ROUND(AVG(annual_revenue), 0) as avg_revenue
FROM listings
GROUP BY neighbourhood_cleansed
ORDER BY avg_revenue DESC
```

| neighbourhood | listings | avg_adr | avg_occupancy | avg_revenue |
|---------------|----------|---------|---------------|-------------|
| Unincorporated Areas | 8,662 | $189 | 0.42 | $28,763 |
| City of Henderson | 678 | $198 | 0.28 | $19,418 |
| City of North Las Vegas | 649 | $164 | 0.33 | $18,692 |
| Boulder City | 13 | $131 | 0.38 | $17,507 |
| City of Las Vegas | 1,715 | $151 | 0.32 | $17,080 |
| City of Mesquite | 110 | $174 | 0.24 | $14,890 |
| Nellis AFB | 6 | $92 | 0.11 | $4,778 |

**Key Insight:** Unincorporated Areas (the Strip) dominates with highest revenue despite not having the highest ADR. The 42% occupancy rate drives superior returns.

### Revenue by Bedroom Count (Strip Only)
```sql
SELECT 
    bedrooms,
    COUNT(*) as listings,
    ROUND(AVG(price_clean), 0) as avg_adr,
    ROUND(AVG(occupancy_rate), 2) as avg_occupancy,
    ROUND(AVG(annual_revenue), 0) as avg_revenue
FROM listings
WHERE neighbourhood_cleansed = 'Unincorporated Areas'
  AND bedrooms BETWEEN 1 AND 6
GROUP BY bedrooms
ORDER BY bedrooms
```

| bedrooms | listings | avg_adr | avg_occupancy | avg_revenue |
|----------|----------|---------|---------------|-------------|
| 1 | 3,869 | $135 | 0.42 | $21,065 |
| 2 | 1,626 | $221 | 0.45 | $35,746 |
| 3 | 976 | $205 | 0.45 | $33,033 |
| 4 | 775 | $250 | 0.40 | $34,983 |
| 5 | 458 | $339 | 0.41 | $48,035 |
| 6 | 121 | $444 | 0.37 | $62,479 |

___

# Occupancy & Revenue Calculation <a name="occupancy-revenue"></a>

The calendar dataset contained 6.4 million records showing daily availability for each listing. We used this to calculate true occupancy rates.

### Understanding the Calendar Data
```python
# Check availability values
calendar['available'].value_counts()

# Results:
# t    3,516,536  (available)
# f    2,916,225  (booked/unavailable)
```

In Airbnb calendar data:
* `t` = available (not booked)
* `f` = unavailable (booked or blocked)

### Calculating Occupancy per Listing
```python
# Calculate occupancy rate per listing
# 'f' means unavailable (booked), so we calculate percentage of 'f' values
occupancy = calendar.groupby('listing_id')['available'].apply(lambda x: (x == 'f').mean())
occupancy = occupancy.reset_index()
occupancy.columns = ['id', 'occupancy_rate']

print(f"Occupancy rates calculated for {len(occupancy):,} listings")
print(f"Average occupancy: {occupancy['occupancy_rate'].mean() * 100:.1f}%")

# Results:
# Occupancy rates calculated for 17,624 listings
# Average occupancy: 45.3%
```

### Merging and Revenue Calculation
```python
# Merge occupancy with listings
listings_full = listings_clean.merge(occupancy, on='id')

# Calculate annual revenue (ADR x Occupancy x 365)
listings_full['annual_revenue'] = listings_full['price_clean'] * listings_full['occupancy_rate'] * 365

print(f"Listings with revenue: {len(listings_full):,}")
print(f"Average annual revenue: ${listings_full['annual_revenue'].mean():,.0f}")

# Results:
# Listings with revenue: 11,833
# Average annual revenue: $25,828
```

### Why Percentiles Matter

Average revenue ($25,828) includes poorly managed properties. Top operators achieve significantly more:
```python
# Calculate percentile performance
airbnb_filtered = listings_full[
    (listings_full['neighbourhood_cleansed'] == 'Unincorporated Areas') &
    (listings_full['bedrooms'].between(1, 6))
]

# 75th percentile (good operators)
top_75 = airbnb_filtered.groupby('bedrooms')['annual_revenue'].quantile(0.75)

# 90th percentile (excellent operators)
top_90 = airbnb_filtered.groupby('bedrooms')['annual_revenue'].quantile(0.90)
```

| bedrooms | revenue_75th | revenue_90th |
|----------|--------------|--------------|
| 1 | $27,664 | $54,612 |
| 2 | $50,322 | $86,625 |
| 3 | $42,685 | $66,386 |
| 4 | $50,946 | $73,412 |
| 5 | $65,182 | $107,831 |
| 6 | $92,225 | $148,000 |

___

# ROI Analysis <a name="roi-analysis"></a>

The core analysis: combining Airbnb revenue with MLS home prices to calculate true ROI.

### MLS Price Analysis
```sql
SELECT 
    "Beds Total" as bedrooms,
    COUNT(*) as homes_sold,
    ROUND(AVG(price_clean), 0) as avg_price,
    ROUND(MIN(price_clean), 0) as min_price,
    ROUND(MAX(price_clean), 0) as max_price
FROM mls
GROUP BY "Beds Total"
ORDER BY bedrooms
```

| bedrooms | homes_sold | avg_price | min_price | max_price |
|----------|------------|-----------|-----------|-----------|
| 1 | 18 | $277,504 | $80,000 | $734,573 |
| 2 | 314 | $361,188 | $121,000 | $500,000 |
| 3 | 1,636 | $403,568 | $107,800 | $500,000 |
| 4 | 730 | $425,455 | $61,750 | $500,000 |
| 5 | 245 | $524,622 | $215,000 | $660,000 |
| 6 | 47 | $667,422 | $310,000 | $850,000 |

### ROI Calculation
```sql
SELECT 
    t.bedrooms,
    ROUND(t.revenue_75th, 0) as revenue_75th,
    ROUND(t.revenue_90th, 0) as revenue_90th,
    ROUND(m.avg_price, 0) as home_price,
    ROUND(m.avg_price * 0.06, 0) as annual_mortgage,
    ROUND(t.revenue_75th - (m.avg_price * 0.06), 0) as net_income_75th,
    ROUND(t.revenue_90th - (m.avg_price * 0.06), 0) as net_income_90th,
    ROUND(((t.revenue_75th - (m.avg_price * 0.06)) / m.avg_price) * 100, 2) as roi_75th,
    ROUND(((t.revenue_90th - (m.avg_price * 0.06)) / m.avg_price) * 100, 2) as roi_90th
FROM top_performers t
JOIN (
    SELECT "Beds Total" as bedrooms, AVG(price_clean) as avg_price
    FROM mls
    GROUP BY "Beds Total"
) m ON t.bedrooms = m.bedrooms
ORDER BY roi_90th DESC
```

| bedrooms | revenue_75th | revenue_90th | home_price | annual_mortgage | net_income_75th | net_income_90th | roi_75th | roi_90th |
|----------|--------------|--------------|------------|-----------------|-----------------|-----------------|----------|----------|
| 2 | $50,322 | $86,625 | $361,188 | $21,671 | $28,650 | $64,954 | 7.93% | 17.98% |
| 6 | $92,225 | $148,000 | $667,422 | $40,045 | $52,180 | $107,955 | 7.82% | 16.17% |
| 5 | $65,182 | $107,831 | $524,622 | $31,477 | $33,705 | $76,353 | 6.42% | 14.55% |
| 1 | $27,664 | $54,612 | $277,504 | $16,650 | $11,014 | $37,962 | 3.97% | 13.68% |
| 4 | $50,946 | $73,412 | $425,455 | $25,527 | $25,418 | $47,885 | 5.97% | 11.25% |
| 3 | $42,685 | $66,386 | $403,568 | $24,214 | $18,471 | $42,172 | 4.58% | 10.45% |

**Key Finding:** 2-bedroom properties deliver the best ROI (18%) due to optimal balance of revenue potential and purchase price.

___

# Amenity Impact Analysis <a name="amenity-analysis"></a>

We analyzed which amenities actually drive bookings versus being merely "nice to have."

### Identifying High-Value Amenities
```python
# Parse amenities from string format
import ast
listings_full['amenities_list'] = listings_full['amenities'].apply(ast.literal_eval)

# Define high-value amenities to analyze
high_value = ['pool', 'hot tub', 'sauna', 'theater', 'arcade', 'pool table', 
              'game', 'gym', 'bbq', 'grill', 'fire pit', 'outdoor kitchen', 
              'golf', 'view', 'spa']

# Check if listing has each amenity
def has_amenity(amenities_list, keyword):
    return any(keyword.lower() in a.lower() for a in amenities_list)

for amenity in high_value:
    col_name = amenity.replace(' ', '_')
    listings_full[col_name] = listings_full['amenities_list'].apply(
        lambda x: has_amenity(x, amenity)
    )
```

### Revenue Impact Calculation
```python
# Compare revenue for each amenity
amenity_impact = []

for amenity in high_value:
    col_name = amenity.replace(' ', '_')
    with_count = listings_full[col_name].sum()
    
    if with_count > 0:
        without = listings_full[listings_full[col_name] == False]['annual_revenue'].mean()
        with_amenity = listings_full[listings_full[col_name] == True]['annual_revenue'].mean()
        diff = with_amenity - without
        pct_lift = (diff / without) * 100
        
        amenity_impact.append({
            'amenity': amenity,
            'listings_with': int(with_count),
            'revenue_without': round(without, 0),
            'revenue_with': round(with_amenity, 0),
            'revenue_lift': round(diff, 0),
            'pct_lift': round(pct_lift, 1)
        })

amenity_df = pd.DataFrame(amenity_impact).sort_values('pct_lift', ascending=False)
```

### Results

| amenity | listings_with | revenue_without | revenue_with | revenue_lift | pct_lift |
|---------|---------------|-----------------|--------------|--------------|----------|
| pool | 7,296 | $16,410 | $31,685 | $15,274 | 93.1% |
| sauna | 489 | $24,993 | $45,214 | $20,221 | 80.9% |
| hot tub | 4,813 | $20,266 | $33,940 | $13,674 | 67.5% |
| pool table | 1,742 | $23,531 | $39,135 | $15,604 | 66.3% |
| arcade | 653 | $25,000 | $40,014 | $15,015 | 60.1% |
| theater | 243 | $25,585 | $37,432 | $11,847 | 46.3% |
| outdoor kitchen | 399 | $25,495 | $35,367 | $9,872 | 38.7% |
| game | 2,077 | $24,399 | $32,539 | $8,140 | 33.4% |
| golf | 289 | $25,640 | $33,346 | $7,706 | 30.1% |
| bbq | 4,686 | $23,310 | $29,669 | $6,359 | 27.3% |

**Key Finding:** Pools nearly double revenue (+93%). Hot tubs, saunas, and entertainment amenities (pool tables, arcades) also provide significant lifts.

___

# Hotel Price Scraping <a name="hotel-scraping"></a>

To benchmark Airbnb against hotels, we scraped pricing data via the Booking.com API.

### API Setup
```python
import requests
from datetime import datetime, timedelta
import time

API_KEY = "your_api_key_here"
API_HOST = "booking-com15.p.rapidapi.com"

headers = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}

# Get Las Vegas destination ID
url = "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchDestination"
params = {"query": "Las Vegas"}

response = requests.get(url, headers=headers, params=params)
data = response.json()

# Result: dest_id = "1704" (Las Vegas Strip, 420 hotels)
```

### Multi-Date Scraping Strategy

To get representative pricing, we scraped 8 date ranges across 3 months:
```python
# Generate date ranges: 4 full weeks + 4 weekends
date_ranges = []

# Full weeks (Mon-Sun, 7 nights)
for weeks_out in [2, 6, 10, 14]:
    start = datetime.now() + timedelta(weeks=weeks_out)
    start = start - timedelta(days=start.weekday())  # Adjust to Monday
    end = start + timedelta(days=7)
    date_ranges.append({
        'type': 'full_week',
        'checkin': start.strftime('%Y-%m-%d'),
        'checkout': end.strftime('%Y-%m-%d'),
        'nights': 7
    })

# Weekends (Fri-Sun, 2 nights)
for weeks_out in [3, 7, 11, 15]:
    start = datetime.now() + timedelta(weeks=weeks_out)
    start = start + timedelta(days=(4 - start.weekday()) % 7)  # Adjust to Friday
    end = start + timedelta(days=2)
    date_ranges.append({
        'type': 'weekend',
        'checkin': start.strftime('%Y-%m-%d'),
        'checkout': end.strftime('%Y-%m-%d'),
        'nights': 2
    })
```

### Fetching Hotel Data
```python
url = "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchHotels"
all_results = []

for dr in date_ranges:
    print(f"Fetching {dr['type']} {dr['checkin']}...", end=" ")
    
    params = {
        "dest_id": "1704",
        "search_type": "district",
        "arrival_date": dr['checkin'],
        "departure_date": dr['checkout'],
        "adults": 2,
        "room_qty": 1,
        "currency_code": "USD"
    }
    
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    hotels = data.get('data', {}).get('hotels', [])
    
    for hotel in hotels:
        prop = hotel.get('property', {})
        total_price = prop.get('priceBreakdown', {}).get('grossPrice', {}).get('value', 0)
        nightly_rate = total_price / dr['nights'] if dr['nights'] > 0 else 0
        
        all_results.append({
            'hotel': prop.get('name', 'Unknown'),
            'stars': prop.get('accuratePropertyClass', 0),
            'review_score': prop.get('reviewScore', 0),
            'date_type': dr['type'],
            'checkin': dr['checkin'],
            'nights': dr['nights'],
            'total_price': total_price,
            'nightly_rate': nightly_rate
        })
    
    print(f"{len(hotels)} hotels")
    time.sleep(1)  # Respect API rate limits

# Results: 160 hotel records across 8 date ranges
```

### Hotel Price Analysis by Star Rating
```python
star_analysis = hotel_df.groupby('stars').agg({
    'nightly_rate': ['mean', 'median', 'count'],
    'review_score': 'mean'
}).round(0)
```

| stars | avg_rate | median_rate | count | avg_review |
|-------|----------|-------------|-------|------------|
| 2 | $119 | $119 | 2 | 8.0 |
| 3 | $180 | $169 | 62 | 8.0 |
| 4 | $488 | $269 | 54 | 8.0 |
| 5 | $525 | $553 | 42 | 9.0 |

___

# Hotel vs Airbnb Comparison <a name="hotel-comparison"></a>

We compared hotels by star rating to Airbnb by performance percentile.

### Building the Comparison
```python
# Hotel medians by stars
hotel_3star = hotel_df[hotel_df['stars'] == 3]['nightly_rate'].median()
hotel_4star = hotel_df[hotel_df['stars'] == 4]['nightly_rate'].median()
hotel_5star = hotel_df[hotel_df['stars'] == 5]['nightly_rate'].median()

# Airbnb percentiles (entire homes, Unincorporated Areas)
airbnb_filtered = listings_full[
    (listings_full['neighbourhood_cleansed'] == 'Unincorporated Areas') &
    (listings_full['room_type'] == 'Entire home/apt')
]

airbnb_median = airbnb_filtered['price_clean'].median()
airbnb_75th = airbnb_filtered['price_clean'].quantile(0.75)
airbnb_90th = airbnb_filtered['price_clean'].quantile(0.90)
```

### Results

| Option | Nightly Rate | Kitchen |
|--------|--------------|---------|
| 3-Star Hotel | $169 | No |
| 4-Star Hotel | $269 | No |
| 5-Star Hotel | $553 | No |
| Airbnb Entire Home (Median) | $163 | Yes |
| Airbnb Entire Home (75th %) | $249 | Yes |
| Airbnb Entire Home (90th %) | $389 | Yes |

### Guest Value Proposition

| Tier | Hotel | Airbnb | Savings |
|------|-------|--------|---------|
| Budget | $169 (3-star) | $163 | $6 (4%) |
| Mid-Range | $269 (4-star) | $249 | $20 (7%) |
| Premium | $553 (5-star) | $389 | $164 (30%) |

**Key Finding:** Airbnb offers 10-30% savings versus comparable hotels, plus full kitchens, more space, and often private pools.

___

# Visualizations <a name="visualizations"></a>

### ROI by Bedroom Count
```python
fig, ax = plt.subplots(figsize=(10, 6))

x = roi_analysis['bedrooms']
width = 0.35

bars1 = ax.bar(x - width/2, roi_analysis['roi_75th'], width, 
               label='Good Operator (75th %)', color='#3498db')
bars2 = ax.bar(x + width/2, roi_analysis['roi_90th'], width, 
               label='Excellent Operator (90th %)', color='#e67e22')

ax.set_xlabel('Bedrooms')
ax.set_ylabel('ROI %')
ax.set_title('ROI by Bedroom Count (Unincorporated Areas)')
ax.legend()

plt.tight_layout()
plt.show()
```

![ROI by Bedroom Count](/img/posts/roi_by_bedroom.png "ROI by Bedroom Count")

### Hotel vs Airbnb Comparison
```python
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Budget', 'Mid-Range', 'Premium']
hotel_prices = [hotel_3star, hotel_4star, hotel_5star]
airbnb_prices = [airbnb_median, airbnb_75th, airbnb_90th]

x = range(len(categories))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], hotel_prices, width, 
               label='Hotel (No Kitchen)', color='#e74c3c')
bars2 = ax.bar([i + width/2 for i in x], airbnb_prices, width, 
               label='Airbnb (Full Kitchen)', color='#3498db')

ax.set_ylabel('Nightly Rate ($)')
ax.set_title('Hotel vs Airbnb: Nightly Rate Comparison (Las Vegas Strip)')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

plt.tight_layout()
plt.show()
```

![Hotel vs Airbnb Comparison](/img/posts/hotel_vs_airbnb.png "Hotel vs Airbnb Comparison")

### Top Amenities by Revenue Impact
```python
top_10 = amenity_df.head(10)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top_10['amenity'], top_10['pct_lift'], color='#2ecc71')

ax.set_xlabel('Revenue Lift (%)')
ax.set_title('Top 10 Amenities by Revenue Impact')
ax.invert_yaxis()

for bar, pct in zip(bars, top_10['pct_lift']):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
            f'{pct:.0f}%', va='center')

plt.tight_layout()
plt.show()
```

![Top Amenities by Revenue Impact](/img/posts/amenity_impact.png "Top Amenities by Revenue Impact")

___

# Investment Recommendations <a name="recommendations"></a>

Based on our comprehensive analysis, here are prioritized investment recommendations:

### Best Bets (High ROI + Reliable Data)

** 1st Choice: 5-Bedroom in Unincorporated Areas **
* Home Price: ~$525K
* Expected Revenue: $65K - $108K/year
* Net Income: $34K - $76K/year
* ROI: 6.4% - 14.6%
* Why: Higher cash flow, good for groups/events, solid data (458 listings, 245 home sales)

** 2nd Choice: 6-Bedroom in Unincorporated Areas **
* Home Price: ~$667K
* Expected Revenue: $92K - $148K/year
* Net Income: $52K - $108K/year
* ROI: 7.8% - 16.2%
* Why: Highest total income, premium market (121 listings, 47 home sales), not 1st because there is less data here and they are harder to find

** 3rd Choice: 2-Bedroom in Unincorporated Areas (Las Vegas Strip) **
* Home Price: ~$361K
* Expected Revenue: $50K - $87K/year (75th-90th percentile)
* Net Income: $29K - $65K/year
* ROI: 7.9% - 18.0%
* Why: Best ROI (but since I am optimizing for income, this is my 3rd choice), most liquid market, strong data (1,626 Airbnb listings, 314 home sales)



### Must-Have Amenities (in order of impact)

| Priority | Amenity | Revenue Lift | Investment |
|----------|---------|--------------|------------|
| 1 | Pool | +93% | High (existing or install) |
| 2 | Hot Tub | +68% | Medium ($5-15K) |
| 3 | Sauna | +81% | Medium ($3-8K) |
| 4 | Pool Table | +66% | Low ($2-5K) |
| 5 | Arcade Games | +60% | Low ($1-3K) |

### Guest Value Proposition

When marketing to potential guests, emphasize:

* **10-30% savings** versus comparable hotel rooms
* **Full kitchen** - hotels don't offer this
* **More space** - entire homes vs single rooms
* **Private amenities** - pools, hot tubs, game rooms
* **Ideal for:** Families, groups, extended stays, bachelor/bachelorette parties

___

# Growth & Next Steps <a name="growth-next-steps"></a>

### Immediate Actions

The analysis framework is ready for immediate use:

**Property Search Criteria:**
* Location: Unincorporated Areas (Las Vegas Strip)
* Size: 2-bedroom (best ROI) or 5-6 bedroom (best cash flow)
* Must-have: Pool
* Nice-to-have: Hot tub, game room
* Budget: $350-400K (2BR) or $500-700K (5-6BR)

### Model Enhancements

**Additional Data Sources:**
* Seasonal analysis (revenue by month) - identify peak/off-peak pricing
* Expense breakdown (cleaning $100-200/turn, management 20-25%, utilities $200-400/mo)
* Competitor pricing via PriceLabs or Wheelhouse integration

**Advanced Analytics:**
* Price elasticity by property type
* Optimal pricing model based on occupancy targets
* Demand forecasting for inventory management

### Technology Improvements

**Interactive Dashboard:**
* Build Streamlit app for real-time property evaluation
* Input address → output expected revenue, ROI, comp analysis
* Integrate Zillow API for automated home valuations

**Automation:**
* Scheduled data refresh from Inside Airbnb (quarterly)
* MLS integration for live home listings
* Alert system for properties meeting criteria

### Market Expansion

Apply this framework to other high-potential markets:
* Phoenix, AZ
* Austin, TX
* Miami, FL
* Nashville, TN
* Scottsdale, AZ

Each market would require:
* Inside Airbnb data download
* MLS access or Redfin/Zillow data
* Local hotel pricing via API
* Market-specific amenity analysis

___

### Database Schema

Final SQLite database contains 7 tables for future analysis:

| Table | Rows | Description |
|-------|------|-------------|
| listings | 11,833 | Cleaned Airbnb listings with revenue |
| mls | 2,990 | Home sales by bedroom count |
| neighbourhoods | 7 | Geographic areas |
| top_performers | 6 | Revenue by bedroom (75th & 90th %) |
| amenity_impact | 15 | Amenity revenue analysis |
| hotels | 160 | Scraped hotel pricing |
| hotel_vs_airbnb | 6 | Competitive comparison |

___

### What I Learned

* **Percentiles > Averages:** Poorly managed properties skew averages; use 75th/90th percentile for realistic projections
* **Occupancy drives revenue:** Henderson had highest ADR but lower revenue due to 28% vs 42% occupancy
* **Amenities matter:** Pool alone adds 93% revenue lift - worth premium in purchase price
* **API integration:** RapidAPI provides accessible hotel pricing data for competitive analysis
* **SQL + Python:** Combining SQL for complex queries with Python for analysis/visualization is powerful

___

*This project demonstrates end-to-end data science capabilities from data collection and cleaning through SQL analysis, API integration, and business recommendations—showing how analytics can drive real estate investment decisions with quantified ROI projections.*

[View the full code on GitHub](https://github.com/yourusername/vegas-airbnb-analysis)
