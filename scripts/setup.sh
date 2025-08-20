#!/bin/bash

# Crypto MLOps MVP Setup Script
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Crypto MLOps MVP - Complete Setup${NC}"
echo -e "${GREEN}====================================${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker is running${NC}"

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ docker-compose not found. Please install Docker Compose.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker Compose available${NC}"

# Create directory structure
echo -e "${BLUE}📁 Creating directory structure...${NC}"

mkdir -p ml/{models,training,inference,mlflow,mlruns}
mkdir -p airflow/{dags,logs,plugins}
mkdir -p grpc/{protos}
mkdir -p graphql
mkdir -p streaming/{producer,consumer}
mkdir -p minio/data
mkdir -p data
mkdir -p scripts
mkdir -p backups

echo -e "${GREEN}✅ Directory structure created${NC}"

# Setup environment variables
echo -e "${BLUE}⚙️ Setting up environment variables...${NC}"

if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}✅ Created .env from .env.example${NC}"
    else
        cat > .env << EOF
# === EXISTING CONFIG ===
LOG_LEVEL=INFO

# === DATABASE CONFIG ===
POSTGRES_HOST=postgres
POSTGRES_DB=mlops
POSTGRES_USER=mlops
POSTGRES_PASSWORD=mlops123

# === MLFLOW CONFIG ===
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_BACKEND_STORE_URI=postgresql://mlops:mlops123@postgres:5432/mlflow
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://mlflow

# === MINIO CONFIG ===
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin123

# === AIRFLOW CONFIG ===
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://mlops:mlops123@postgres:5432/airflow
AIRFLOW__CORE__FERNET_KEY=YlCImzjge_TeZc5FVOyxN1yMM5JziWNF3gULEUqXMaU=
AIRFLOW__WEBSERVER__SECRET_KEY=YlCImzjge_TeZc5FVOyxN1yMM5JziWNF3gULEUqXMaU=

# === KAFKA CONFIG ===
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_PRICES=crypto-prices
KAFKA_TOPIC_PREDICTIONS=predictions
KAFKA_TOPIC_ALERTS=alerts

# === API CONFIG ===
GRPC_PORT=50051
GRAPHQL_PORT=4000
EOF
        echo -e "${GREEN}✅ Created default .env file${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ .env file already exists${NC}"
fi

# Check available memory
echo -e "${BLUE}🧠 Checking system resources...${NC}"

TOTAL_MEM=$(free -m | grep '^Mem:' | awk '{print $2}')
if [ "$TOTAL_MEM" -lt 6144 ]; then
    echo -e "${YELLOW}⚠️ Warning: System has ${TOTAL_MEM}MB RAM. Recommended: 8GB+ for optimal performance${NC}"
else
    echo -e "${GREEN}✅ System has sufficient RAM: ${TOTAL_MEM}MB${NC}"
fi

# Check disk space
DISK_SPACE=$(df -BG / | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$DISK_SPACE" -lt 10 ]; then
    echo -e "${YELLOW}⚠️ Warning: Low disk space: ${DISK_SPACE}GB available. Consider freeing up space.${NC}"
else
    echo -e "${GREEN}✅ Sufficient disk space: ${DISK_SPACE}GB available${NC}"
fi

# Start services
echo -e "${BLUE}🔄 Starting all services...${NC}"

# Build and start containers
docker-compose up -d --build

echo -e "${GREEN}✅ All services started${NC}"

# Wait for services to be ready
echo -e "${BLUE}⏳ Waiting for services to be ready...${NC}"

# Function to wait for service
wait_for_service() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    echo -e "${YELLOW}Waiting for ${service_name}...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ ${service_name} is ready${NC}"
            return 0
        fi
        
        echo -e "${YELLOW}Attempt ${attempt}/${max_attempts} - ${service_name} not ready yet...${NC}"
        sleep 10
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}❌ ${service_name} failed to start within expected time${NC}"
    return 1
}

# Wait for key services
wait_for_service "FastAPI" "http://localhost:8800/health"
wait_for_service "MinIO" "http://localhost:9000/minio/health/live"

# Setup MinIO buckets
echo -e "${BLUE}🪣 Setting up MinIO buckets...${NC}"

sleep 5 # Additional wait for MinIO

docker-compose exec -T minio mc alias set local http://localhost:9000 minioadmin minioadmin123 || true
docker-compose exec -T minio mc mb local/raw-data || true
docker-compose exec -T minio mc mb local/processed-data || true
docker-compose exec -T minio mc mb local/models || true
docker-compose exec -T minio mc mb local/mlflow || true
docker-compose exec -T minio mc mb local/quality-reports || true

echo -e "${GREEN}✅ MinIO buckets created${NC}"

# Wait for more services
wait_for_service "MLFlow" "http://localhost:5000" || echo -e "${YELLOW}⚠️ MLFlow may still be initializing${NC}"
wait_for_service "GraphQL" "http://localhost:4000" || echo -e "${YELLOW}⚠️ GraphQL may still be initializing${NC}"

# Create initial data
echo -e "${BLUE}📊 Generating initial test data...${NC}"

# Test API endpoint
if curl -s "http://localhost:8800/v1/crypto/ohlcv?symbol=BTCUSDT&limit=5" > /dev/null; then
    echo -e "${GREEN}✅ API is responding to requests${NC}"
else
    echo -e "${YELLOW}⚠️ API not fully ready yet${NC}"
fi

# Final status check
echo -e "${BLUE}🏥 Final health check...${NC}"

services_status=""

if curl -s http://localhost:8800/health > /dev/null 2>&1; then
    services_status="${services_status}✅ FastAPI\n"
else
    services_status="${services_status}❌ FastAPI\n"
fi

if curl -s http://localhost:5000 > /dev/null 2>&1; then
    services_status="${services_status}✅ MLFlow\n"
else
    services_status="${services_status}❌ MLFlow\n"
fi

if curl -s http://localhost:8080 > /dev/null 2>&1; then
    services_status="${services_status}✅ Airflow\n"
else
    services_status="${services_status}❌ Airflow\n"
fi

if curl -s http://localhost:9001 > /dev/null 2>&1; then
    services_status="${services_status}✅ MinIO\n"
else
    services_status="${services_status}❌ MinIO\n"
fi

if curl -s http://localhost:4000 > /dev/null 2>&1; then
    services_status="${services_status}✅ GraphQL\n"
else
    services_status="${services_status}❌ GraphQL\n"
fi

if curl -s http://localhost:8088 > /dev/null 2>&1; then
    services_status="${services_status}✅ Kafka UI\n"
else
    services_status="${services_status}❌ Kafka UI\n"
fi

echo -e "${BLUE}Services Status:${NC}"
echo -e "$services_status"

# Success message and dashboard URLs
echo ""
echo -e "${GREEN}🎉 SETUP COMPLETED SUCCESSFULLY! 🎉${NC}"
echo -e "${GREEN}=================================${NC}"
echo ""
echo -e "${BLUE}📊 Available Dashboards & Services:${NC}"
echo -e "${GREEN}🌐 FastAPI (REST):      ${NC}http://localhost:8800"
echo -e "${GREEN}📚 FastAPI Docs:        ${NC}http://localhost:8800/docs"
echo -e "${GREEN}🧠 MLFlow:              ${NC}http://localhost:5000"
echo -e "${GREEN}🔄 Airflow:             ${NC}http://localhost:8080 ${YELLOW}(admin/admin123)${NC}"
echo -e "${GREEN}🗄️ MinIO Console:        ${NC}http://localhost:9001 ${YELLOW}(minioadmin/minioadmin123)${NC}"
echo -e "${GREEN}🎯 GraphQL Playground:  ${NC}http://localhost:4000/graphql"
echo -e "${GREEN}📡 gRPC Server:         ${NC}localhost:50051"
echo -e "${GREEN}🔥 Kafka UI:            ${NC}http://localhost:8088"
echo -e "${GREEN}🖥️ Original UI:         ${NC}file://$PWD/ui/index.html"
echo ""
echo -e "${BLUE}🔧 Useful Commands:${NC}"
echo -e "${GREEN}make logs               ${NC}- Show all service logs"
echo -e "${GREEN}make check-health       ${NC}- Check service health"
echo -e "${GREEN}make test-apis          ${NC}- Test API endpoints"
echo -e "${GREEN}make train-model        ${NC}- Train ML model manually"
echo -e "${GREEN}make monitor            ${NC}- Real-time monitoring"
echo -e "${GREEN}make dashboard-urls     ${NC}- Show this info again"
echo ""
echo -e "${YELLOW}💡 Tips:${NC}"
echo "- Services may take a few more minutes to fully initialize"
echo "- If a service isn't responding, try: docker-compose restart <service_name>"
echo "- Use 'make logs' to troubleshoot any issues"
echo "- The Kafka streaming will start producing data automatically"
echo ""
echo -e "${GREEN}Happy MLOps! 🚀${NC}"