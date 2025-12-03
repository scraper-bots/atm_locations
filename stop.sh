#!/bin/bash

# BOB ATM Dashboard Stop Script
echo "🛑 Stopping Bank of Baku ATM Dashboard..."
echo ""

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Stop containers
$COMPOSE_CMD down

echo ""
echo "✅ Dashboard stopped successfully!"
echo ""
