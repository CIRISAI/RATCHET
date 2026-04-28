#!/bin/bash
# CIRISLens Backup Script
# Run daily via cron: 0 2 * * * /opt/cirislens/scripts/backup.sh

set -e

# Configuration
BACKUP_ROOT="/backups/cirislens"
BACKUP_DIR="$BACKUP_ROOT/$(date +%Y%m%d_%H%M%S)"
RETENTION_DAYS=30
DEPLOY_DIR="/opt/cirislens"

# Create backup directory
mkdir -p $BACKUP_DIR

echo "Starting CIRISLens backup to $BACKUP_DIR"

# Change to deployment directory
cd $DEPLOY_DIR

# 1. Backup PostgreSQL database
echo "Backing up PostgreSQL..."
docker compose -f docker-compose.managed.yml -f docker-compose.prod.yml exec -T cirislens-db \
    pg_dump -U cirislens cirislens | gzip > $BACKUP_DIR/database.sql.gz

# 2. Backup Grafana dashboards and settings
echo "Backing up Grafana..."
docker compose -f docker-compose.managed.yml -f docker-compose.prod.yml exec -T cirislens-grafana \
    tar czf - /var/lib/grafana/dashboards /var/lib/grafana/grafana.db 2>/dev/null > $BACKUP_DIR/grafana.tar.gz || true

# 3. Backup configurations
echo "Backing up configurations..."
tar czf $BACKUP_DIR/configs.tar.gz \
    config/ \
    sql/ \
    dashboards/ \
    nginx/ \
    .env \
    docker-compose.*.yml \
    2>/dev/null || true

# 4. Backup agent tokens and settings from database
echo "Backing up agent configurations..."
docker compose -f docker-compose.managed.yml -f docker-compose.prod.yml exec -T cirislens-db \
    psql -U cirislens -d cirislens -c "\copy (SELECT * FROM agent_otlp_configs) TO STDOUT WITH CSV HEADER" \
    > $BACKUP_DIR/agent_configs.csv

# 5. Create backup manifest
echo "Creating manifest..."
cat > $BACKUP_DIR/manifest.txt << EOF
CIRISLens Backup Manifest
========================
Date: $(date)
Hostname: $(hostname)
Directory: $BACKUP_DIR

Files:
$(ls -lh $BACKUP_DIR | tail -n +2)

Docker Status:
$(docker compose -f docker-compose.managed.yml -f docker-compose.prod.yml ps --format "table {{.Name}}\t{{.Status}}")

Disk Usage:
$(df -h /data/cirislens)
EOF

# 6. Compress entire backup
echo "Compressing backup..."
cd $BACKUP_ROOT
tar czf "cirislens_$(date +%Y%m%d_%H%M%S).tar.gz" $(basename $BACKUP_DIR)
rm -rf $BACKUP_DIR

# 7. Clean old backups
echo "Cleaning old backups (keeping $RETENTION_DAYS days)..."
find $BACKUP_ROOT -name "cirislens_*.tar.gz" -type f -mtime +$RETENTION_DAYS -delete

# 8. Report
echo "Backup complete!"
echo "Current backups:"
ls -lh $BACKUP_ROOT/cirislens_*.tar.gz | tail -5

# Optional: Upload to S3 or remote storage
# aws s3 cp $BACKUP_ROOT/cirislens_$(date +%Y%m%d)*.tar.gz s3://your-bucket/cirislens-backups/