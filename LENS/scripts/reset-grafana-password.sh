#!/bin/bash
# Reset Grafana admin password in production
# Usage: ./reset-grafana-password.sh [new-password]

set -e

echo "Grafana Admin Password Reset Tool"
echo "================================="

# Get new password from argument or prompt
if [ -z "$1" ]; then
    echo -n "Enter new admin password: "
    read -s NEW_PASSWORD
    echo
    echo -n "Confirm new admin password: "
    read -s CONFIRM_PASSWORD
    echo
    
    if [ "$NEW_PASSWORD" != "$CONFIRM_PASSWORD" ]; then
        echo "Passwords do not match!"
        exit 1
    fi
else
    NEW_PASSWORD="$1"
fi

if [ -z "$NEW_PASSWORD" ]; then
    echo "Password cannot be empty!"
    exit 1
fi

# Method 1: Try using Grafana CLI (preferred)
echo "Method 1: Using Grafana CLI..."
if docker compose -f docker-compose.managed.yml -f docker-compose.prod.yml exec cirislens-grafana grafana-cli admin reset-admin-password "$NEW_PASSWORD" 2>/dev/null; then
    echo "✅ Password reset successfully using Grafana CLI"
    exit 0
fi

# Method 2: Direct database update
echo "Method 1 failed, trying Method 2: Direct database update..."

# Hash the new password (Grafana uses bcrypt)
# Note: This requires python3 and bcrypt library
HASHED_PASSWORD=$(python3 -c "
import bcrypt
password = '$NEW_PASSWORD'.encode('utf-8')
salt = bcrypt.gensalt()
hashed = bcrypt.hashpw(password, salt)
print(hashed.decode('utf-8'))
" 2>/dev/null) || {
    echo "Failed to hash password. Installing bcrypt..."
    pip3 install bcrypt --quiet
    HASHED_PASSWORD=$(python3 -c "
import bcrypt
password = '$NEW_PASSWORD'.encode('utf-8')
salt = bcrypt.gensalt()
hashed = bcrypt.hashpw(password, salt)
print(hashed.decode('utf-8'))
")
}

if [ -z "$HASHED_PASSWORD" ]; then
    echo "Failed to generate password hash"
    exit 1
fi

# Update the database directly
echo "Updating Grafana database..."
docker compose -f docker-compose.managed.yml -f docker-compose.prod.yml exec cirislens-grafana sqlite3 /var/lib/grafana/grafana.db "
UPDATE user 
SET password = '$HASHED_PASSWORD', 
    salt = '', 
    rands = '', 
    is_admin = 1 
WHERE login = 'admin';
" 2>/dev/null || {
    # If SQLite fails, try using the Grafana API
    echo "Database update failed, trying API method..."
}

# Method 3: Use the API with temporary default password
echo "Method 3: Using Grafana API..."

# First, we need to get the container to use default credentials temporarily
docker compose -f docker-compose.managed.yml -f docker-compose.prod.yml exec cirislens-grafana sh -c "
    # Try to login with current admin/admin
    curl -s -X POST http://localhost:3000/api/user/password \
        -H 'Content-Type: application/json' \
        -u admin:admin \
        -d '{\"oldPassword\":\"admin\",\"newPassword\":\"$NEW_PASSWORD\"}' && echo 'Password changed via API'
"

# Method 4: Nuclear option - reset the database
echo ""
echo "If all methods failed, you can reset Grafana's database:"
echo "1. Stop Grafana: docker compose down cirislens-grafana"
echo "2. Remove database: docker compose exec cirislens-grafana rm /var/lib/grafana/grafana.db"
echo "3. Update .env file with: GF_ADMIN_PASSWORD=$NEW_PASSWORD"
echo "4. Restart Grafana: docker compose up -d cirislens-grafana"
echo ""

# Test the new password
echo "Testing new password..."
TEST_RESULT=$(curl -s -o /dev/null -w "%{http_code}" -u "admin:$NEW_PASSWORD" http://localhost:3001/api/admin/settings 2>/dev/null)

if [ "$TEST_RESULT" = "200" ]; then
    echo "✅ Password reset successful! You can now login with:"
    echo "   Username: admin"
    echo "   Password: [your new password]"
    
    # Update .env file if it exists
    if [ -f .env ]; then
        echo ""
        echo "Updating .env file..."
        if grep -q "GF_ADMIN_PASSWORD=" .env; then
            sed -i.bak "s/GF_ADMIN_PASSWORD=.*/GF_ADMIN_PASSWORD=$NEW_PASSWORD/" .env
        else
            echo "GF_ADMIN_PASSWORD=$NEW_PASSWORD" >> .env
        fi
        echo "✅ .env file updated"
    fi
else
    echo "⚠️ Password may have been changed but could not verify"
    echo "Try logging in manually at https://agents.ciris.ai/lens/login"
fi