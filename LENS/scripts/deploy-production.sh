#!/bin/bash
# CIRISLens Production Deployment Script
# Deploys CIRISLens on agents.ciris.ai

set -e

echo "CIRISLens Production Deployment Script"
echo "======================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root"
    exit 1
fi

# Configuration
DEPLOY_DIR="/opt/cirislens"
DATA_DIR="/data/cirislens"
NGINX_SITES="/etc/nginx/sites-available"
NGINX_ENABLED="/etc/nginx/sites-enabled"

echo "1. Creating directories..."
mkdir -p $DEPLOY_DIR
mkdir -p $DATA_DIR/{postgres,grafana,prometheus,loki,tempo,mimir,minio}
chmod 755 $DATA_DIR

echo "2. Cloning repository..."
if [ ! -d "$DEPLOY_DIR/.git" ]; then
    git clone https://github.com/CIRISAI/CIRISLens.git $DEPLOY_DIR
else
    cd $DEPLOY_DIR
    git pull origin main
fi

cd $DEPLOY_DIR

echo "3. Setting up environment..."
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << 'EOF'
# Production Configuration
ENV=production
DATABASE_URL=postgresql://cirislens:$(openssl rand -hex 16)@cirislens-db:5432/cirislens

# OAuth Configuration
GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID:-your-client-id}
GOOGLE_CLIENT_SECRET=${GOOGLE_CLIENT_SECRET:-your-secret}
OAUTH_CALLBACK_URL=https://agents.ciris.ai/lens/admin/auth/callback
ALLOWED_DOMAIN=ciris.ai
SESSION_SECRET=$(openssl rand -hex 32)

# Collection Settings
COLLECTION_INTERVAL_SECONDS=30
OTLP_COLLECTION_ENABLED=true

# Grafana Admin (IMPORTANT: Change this!)
GF_ADMIN_USER=admin
GF_ADMIN_PASSWORD=$(openssl rand -hex 16)
GF_ADMIN_EMAIL=admin@ciris.ai
GF_SECRET_KEY=$(openssl rand -hex 32)

# Grafana OAuth (uses same Google OAuth as Admin UI)
# This enables Google Sign-In for Grafana dashboards
GF_AUTH_GOOGLE_ENABLED=true

# MinIO
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=$(openssl rand -hex 16)
EOF
    echo "Please edit .env and add your Google OAuth credentials and agent tokens"
    echo "Press enter to continue after editing..."
    read
fi

echo "4. Installing nginx configuration..."
if [ -f "nginx/lens-site.conf" ]; then
    cp nginx/lens-site.conf $NGINX_SITES/cirislens.conf
    ln -sf $NGINX_SITES/cirislens.conf $NGINX_ENABLED/cirislens.conf
    echo "Testing nginx configuration..."
    nginx -t
    if [ $? -eq 0 ]; then
        echo "Reloading nginx..."
        nginx -s reload
    else
        echo "Nginx configuration error! Please check manually."
        exit 1
    fi
fi

echo "5. Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "Docker not found! Please install Docker first."
    exit 1
fi

if ! docker network ls | grep -q "ciris-network"; then
    echo "Warning: ciris-network not found. Creating it..."
    docker network create ciris-network
fi

echo "6. Starting CIRISLens stack..."
docker compose pull
docker compose up -d

echo "7. Waiting for services to start..."
sleep 15

echo "8. Checking service health..."
docker compose ps

echo "9. Configuring TimescaleDB..."
# Wait for PostgreSQL to be ready
for i in {1..30}; do
    if docker exec cirislens-db pg_isready -U cirislens > /dev/null 2>&1; then
        echo "Database is ready"
        break
    fi
    echo "Waiting for database..."
    sleep 2
done

# Enable TimescaleDB extension if not already enabled
docker exec cirislens-db psql -U cirislens -d cirislens -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;" 2>/dev/null || true

# Check if shared_preload_libraries is configured
PRELOAD=$(docker exec cirislens-db psql -U cirislens -d cirislens -t -c "SHOW shared_preload_libraries;" 2>/dev/null | tr -d ' ')
if [[ ! "$PRELOAD" == *"timescaledb"* ]]; then
    echo "Configuring TimescaleDB shared_preload_libraries..."
    docker exec cirislens-db bash -c "echo \"shared_preload_libraries = 'timescaledb'\" >> /var/lib/postgresql/data/postgresql.conf"
    echo "Restarting database with TimescaleDB enabled..."
    docker restart cirislens-db
    sleep 10
fi

# Run TimescaleDB migration if hypertables don't exist
HT_COUNT=$(docker exec cirislens-db psql -U cirislens -d cirislens -t -c "SELECT count(*) FROM timescaledb_information.hypertables WHERE hypertable_schema = 'cirislens';" 2>/dev/null | tr -d ' ')
if [[ "$HT_COUNT" == "0" || -z "$HT_COUNT" ]]; then
    echo "Running TimescaleDB migration..."
    docker exec -i cirislens-db psql -U cirislens -d cirislens < sql/006_timescaledb_migration.sql 2>&1 || echo "Migration may have partial errors (expected for new installs)"
fi

echo "10. Verifying TimescaleDB setup..."
docker exec cirislens-db psql -U cirislens -d cirislens -c "SELECT hypertable_name FROM timescaledb_information.hypertables WHERE hypertable_schema = 'cirislens';" 2>/dev/null || echo "TimescaleDB verification skipped"

echo "11. Setting up Grafana for public access..."
./scripts/setup-grafana-public.sh

echo "12. Testing endpoints..."
echo -n "API Health: "
curl -s http://localhost:8200/health | jq -r '.status' || echo "Failed"

echo -n "Grafana Health: "
curl -s http://localhost:3001/api/health | jq -r '.database' || echo "Failed"

echo ""
echo "Deployment Complete!"
echo "===================="
echo ""
echo "Access URLs:"
echo "- Grafana Dashboards: https://agents.ciris.ai/lens/ (requires @ciris.ai Google login)"
echo "- Admin UI: https://agents.ciris.ai/lens/admin/"
echo "- API Health: https://agents.ciris.ai/lens/api/health"
echo ""
echo "Grafana Admin Credentials (saved in .env):"
grep "GF_ADMIN_USER\|GF_ADMIN_PASSWORD" .env | grep -v SECRET_KEY
echo ""
echo "TimescaleDB Data Retention (automatic):"
echo "- Metrics: 30 days (hourly aggregates kept 90 days, daily kept 1 year)"
echo "- Logs: 14 days"
echo "- Traces: 14 days"
echo "- Compression: After 7 days (90% space savings)"
echo ""
echo "Next Steps:"
echo "1. Verify telemetry collection: docker logs cirislens-api --tail 50"
echo "2. Check agent discovery: docker exec cirislens-db psql -U cirislens -d cirislens -c 'SELECT agent_id, status, last_seen FROM cirislens.discovered_agents ORDER BY last_seen DESC LIMIT 10;'"
echo "3. Configure Grafana dashboards"
echo "4. Set up backup cron job: crontab -e"
echo "   0 2 * * * /opt/cirislens/scripts/backup.sh"
echo ""
echo "To view logs: docker compose logs -f"