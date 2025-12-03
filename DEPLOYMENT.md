# BOB ATM Strategy Dashboard - Deployment Guide

## Overview
Interactive fullstack dashboard for Bank of Baku's ATM placement strategy analysis. Built with Streamlit and Plotly for real-time data visualization and strategic insights.

## Features
- 📊 Real-time market analysis and KPI tracking
- 🗺️ Interactive maps with competitor positioning
- 🎯 Coverage gap identification with 2km radius analysis
- 🏪 Retail partnership opportunity explorer
- 📈 Competitive intelligence and co-location analysis
- 💰 ROI-ranked expansion recommendations

## Prerequisites
- Docker (20.10+)
- Docker Compose (1.29+)
- 2GB RAM minimum
- Port 8501 available

## Quick Start with Docker

### Option 1: Using Docker Compose (Recommended)

```bash
# Clone or navigate to project directory
cd atm_locations

# Build and start the dashboard
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the dashboard
docker-compose down
```

The dashboard will be available at: **http://localhost:8501**

### Option 2: Using Docker directly

```bash
# Build the image
docker build -t bob-atm-dashboard .

# Run the container
docker run -d \
  --name bob-dashboard \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/charts:/app/charts:ro \
  bob-atm-dashboard

# View logs
docker logs -f bob-dashboard

# Stop the container
docker stop bob-dashboard
docker rm bob-dashboard
```

## Local Development (without Docker)

### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

The dashboard will open automatically at: **http://localhost:8501**

## Project Structure

```
atm_locations/
├── app.py                      # Main Streamlit dashboard
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container configuration
├── docker-compose.yml          # Docker Compose orchestration
├── .dockerignore              # Docker ignore patterns
├── data/
│   ├── combined_locations.csv # 4,377 locations dataset
│   ├── kapital_atms.csv       # Individual bank data
│   ├── abb_atms.csv
│   ├── xalq_atms.csv
│   ├── bob_atms.csv
│   ├── bazarstore_branches.csv
│   └── ...
├── charts/                     # Static analysis charts (12 PNGs)
├── scripts/                    # Data collection scripts
│   ├── scrape_kapital.py
│   ├── combine_datasets.py
│   └── analyze_for_bob.py
└── README.md                   # Strategic analysis presentation
```

## Dashboard Pages

### 1. 📊 Overview
- Key metrics: Total ATMs, market share, gap to leader
- Market share comparison bar chart
- Geographic distribution map
- Coverage gap summary

### 2. 🗺️ Interactive Map
- Toggle layers: BOB ATMs, competitor ATMs, retail locations
- Filter by bank
- Hover for location details
- Real-time statistics

### 3. 🎯 Coverage Gaps
- Identifies competitor locations >2km from BOB
- Heatmap by distance and competitor density
- Gap distribution histograms
- Top 20 gaps by strategic importance

### 4. 🏪 Retail Opportunities
- Partnership opportunities at OBA, Bravo, Bazarstore
- Opportunity score based on distance and foot traffic
- Interactive map of top 20 retail locations
- Opportunities by retail chain

### 5. 📈 Competitor Analysis
- Market position comparison
- Geographic density heatmaps
- Co-location matrix (within 500m)
- Network penetration efficiency scatter plot

### 6. 💰 ROI Rankings
- Multi-factor ROI scoring (coverage gap 30%, demand 40%, retail 30%)
- Interactive top-N location selector
- Color-coded by score tier (Excellent/Good/Fair/Low)
- Investment estimates in AZN
- Phased rollout recommendations
- Downloadable CSV rankings

## Configuration

### Environment Variables

```bash
# Streamlit configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### Custom Port

To run on a different port:

```bash
# Docker Compose: Edit docker-compose.yml
ports:
  - "8080:8501"  # Change 8080 to your desired port

# Direct Docker
docker run -p 8080:8501 bob-atm-dashboard

# Local development
streamlit run app.py --server.port 8080
```

### Coverage Radius

Adjust the coverage radius in the sidebar (0.5km - 5km). Default: 2km.

## Deployment Options

### 1. Cloud Deployment - Streamlit Cloud (Free)

```bash
# Push to GitHub
git init
git add .
git commit -m "Initial commit"
git push origin main

# Visit: https://streamlit.io/cloud
# Connect your GitHub repo
# Deploy app.py
```

### 2. Cloud Deployment - Heroku

```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create app
heroku create bob-atm-dashboard

# Deploy
git push heroku main

# Open app
heroku open
```

Add `setup.sh` and `Procfile` for Heroku:

**setup.sh:**
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

**Procfile:**
```
web: sh setup.sh && streamlit run app.py
```

### 3. Cloud Deployment - AWS EC2

```bash
# SSH into EC2 instance
ssh -i your-key.pem ec2-user@your-instance

# Install Docker
sudo yum update -y
sudo yum install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

# Clone repo
git clone your-repo-url
cd atm_locations

# Run with Docker Compose
docker-compose up -d

# Configure security group to allow port 8501
```

### 4. Cloud Deployment - DigitalOcean

```bash
# Create Droplet with Docker pre-installed
# SSH into droplet

# Clone repo
git clone your-repo-url
cd atm_locations

# Run
docker-compose up -d

# Access at: http://your-droplet-ip:8501
```

### 5. On-Premises Server

```bash
# Copy project to server
scp -r atm_locations user@server:/path/to/deploy

# SSH into server
ssh user@server

# Navigate to project
cd /path/to/deploy/atm_locations

# Run with Docker Compose
docker-compose up -d

# Set up systemd service for auto-start
sudo systemctl enable docker
```

## Data Updates

To update the data:

```bash
# Run data collection scripts
cd scripts
python scrape_kapital.py
python scrape_abb.py
# ... run other scrapers

# Combine datasets
python combine_datasets.py

# Regenerate charts (optional)
python analyze_for_bob.py

# Restart dashboard
docker-compose restart
```

The dashboard automatically reloads when CSV files change.

## Performance Optimization

- **Caching**: All data loading is cached with `@st.cache_data`
- **Lazy loading**: Charts generate on-demand per page
- **Volume mounts**: Data/charts mounted as read-only volumes
- **Health checks**: Automatic container health monitoring

## Troubleshooting

### Port already in use
```bash
# Find process using port 8501
lsof -i :8501

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
```

### Container won't start
```bash
# Check logs
docker-compose logs

# Rebuild without cache
docker-compose build --no-cache
docker-compose up
```

### Data not loading
```bash
# Verify data files exist
ls data/combined_locations.csv

# Check file permissions
chmod 644 data/*.csv

# Restart container
docker-compose restart
```

### Memory issues
```bash
# Increase Docker memory limit
# Docker Desktop -> Settings -> Resources -> Memory (increase to 4GB)

# Or add to docker-compose.yml:
services:
  bob-dashboard:
    mem_limit: 2g
```

## Security Considerations

- Dashboard runs on HTTP by default (use reverse proxy for HTTPS)
- Data volumes mounted as read-only
- No database credentials required (CSV-based)
- Consider VPN or IP whitelisting for production
- Use environment variables for sensitive config

## Monitoring

```bash
# View real-time logs
docker-compose logs -f

# Check container health
docker ps

# View resource usage
docker stats bob-atm-dashboard
```

## Backup

```bash
# Backup data directory
tar -czf bob-data-backup-$(date +%Y%m%d).tar.gz data/

# Backup charts
tar -czf bob-charts-backup-$(date +%Y%m%d).tar.gz charts/
```

## Support

For issues or questions:
1. Check logs: `docker-compose logs`
2. Verify data files in `/data` directory
3. Ensure port 8501 is available
4. Check Docker daemon is running

## License

Internal use only - Bank of Baku Strategic Analysis

---

**Dashboard URL**: http://localhost:8501

**Data Sources**: 7 Banks • 3 Retail Chains • 4,377 Locations

**Last Updated**: 2025-12-03
