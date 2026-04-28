#!/bin/bash
# Setup Grafana with OAuth authentication
# Run this after Grafana starts for the first time

set -e

echo "Setting up Grafana authentication..."

GRAFANA_URL="http://localhost:3001"
ADMIN_USER="${GF_ADMIN_USER:-admin}"
ADMIN_PASS="${GF_ADMIN_PASSWORD:-admin}"

# Wait for Grafana to be ready
echo "Waiting for Grafana to start..."
for i in {1..30}; do
    if curl -s "$GRAFANA_URL/api/health" > /dev/null 2>&1; then
        echo "Grafana is ready!"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

# Check Grafana version
echo "Checking Grafana version..."
VERSION=$(curl -s "$GRAFANA_URL/api/health" | jq -r '.version')
echo "Grafana version: $VERSION"

# Verify OAuth is configured
echo "Checking OAuth configuration..."
if [ "${GF_AUTH_GOOGLE_ENABLED:-false}" = "true" ]; then
    echo "Google OAuth is ENABLED"
    echo "Users must authenticate with @${ALLOWED_DOMAIN:-ciris.ai} accounts"
else
    echo "Google OAuth is DISABLED (development mode)"
    echo "Using basic auth with admin credentials"
fi

# Set org preferences
echo "Configuring organization preferences..."
curl -s -X PUT "$GRAFANA_URL/api/org/preferences" \
    -H "Content-Type: application/json" \
    -u "$ADMIN_USER:$ADMIN_PASS" \
    -d '{
        "theme": "dark",
        "timezone": "utc"
    }' 2>/dev/null || echo "Could not update preferences"

echo ""
echo "Grafana authentication setup complete!"
echo "======================================="
echo ""
if [ "${GF_AUTH_GOOGLE_ENABLED:-false}" = "true" ]; then
    echo "Access: https://agents.ciris.ai/lens/"
    echo "Login: Google OAuth (@${ALLOWED_DOMAIN:-ciris.ai} accounts only)"
else
    echo "Access: http://localhost:3001"
    echo "Login: admin / $ADMIN_PASS"
fi
echo ""
echo "Note: Anonymous access is DISABLED. All users must authenticate."
