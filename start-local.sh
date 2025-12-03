#!/bin/bash

# BOB ATM Dashboard Local Development Script
echo "🏦 Starting Bank of Baku ATM Dashboard (Local Development)..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11+"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📚 Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Check if data exists
if [ ! -f "data/combined_locations.csv" ]; then
    echo ""
    echo "⚠️  Warning: data/combined_locations.csv not found!"
    echo "Please run the data collection scripts first:"
    echo "   cd scripts && python combine_datasets.py"
    echo ""
    exit 1
fi

# Start Streamlit
echo ""
echo "🚀 Starting Streamlit dashboard..."
echo ""
echo "✅ Dashboard will open automatically at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
