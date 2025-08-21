# Crypto MLOps MVP - Extended Makefile

.PHONY: help up down restart logs clean test build setup-buckets train-model dashboard-urls check-health

# Colors
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
BLUE=\033[0;34m
NC=\033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)Crypto MLOps MVP - Available Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(BLUE)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

up: ## Start all services
	@echo "$(GREEN)🚀 Starting all MLOps services...$(NC)"
	docker-compose up -d --build
	@echo "$(GREEN)✅ Services started! Use 'make dashboard-urls' to see all endpoints$(NC)"

down: ## Stop all services
	@echo "$(YELLOW)🛑 Stopping all services...$(NC)"
	docker-compose down

restart: down up ## Restart all services

logs: ## Show logs from all services
	docker-compose logs -f

logs-api: ## Show API logs
	docker-compose logs -f api

logs-mlflow: ## Show MLFlow logs
	docker-compose logs -f mlflow

logs-airflow: ## Show Airflow logs
	docker-compose logs -f airflow-webserver airflow-scheduler

logs-kafka: ## Show Kafka logs
	docker-compose logs -f kafka crypto-producer crypto-consumer

build: ## Build all Docker images
	@echo "$(GREEN)🔨 Building all Docker images...$(NC)"
	docker-compose build --no-cache

clean: ## Clean up Docker resources
	@echo "$(YELLOW)🧹 Cleaning up Docker resources...$(NC)"
	docker-compose down -v --remove-orphans
	docker system prune -f
	docker volume prune -f

setup-env: ## Setup environment files
	@echo "$(GREEN)⚙️ Setting up environment...$(NC)"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)✅ Created .env from .env.example$(NC)"; \
	else \
		echo "$(YELLOW)⚠️ .env already exists$(NC)"; \
	fi

setup-buckets: ## Create MinIO buckets
	@echo "$(GREEN)🪣 Setting up MinIO buckets...$(NC)"
	@sleep 5 # Wait for MinIO to be ready
	docker-compose exec -T minio mc alias set local http://localhost:9000 minioadmin minioadmin123 || true
	docker-compose exec -T minio mc mb local/raw-data || true
	docker-compose exec -T minio mc mb local/processed-data || true
	docker-compose exec -T minio mc mb local/models || true
	docker-compose exec -T minio mc mb local/mlflow || true
	docker-compose exec -T minio mc mb local/quality-reports || true
	@echo "$(GREEN)✅ MinIO buckets created$(NC)"

setup: setup-env up setup-buckets ## Complete setup from scratch
	@echo "$(GREEN)🎉 Complete setup finished!$(NC)"
	@make dashboard-urls

train-model: ## Train ML model manually
	@echo "$(GREEN)🤖 Starting model training...$(NC)"
	docker-compose exec api python /app/ml/models/volatility_lstm.py
	@echo "$(GREEN)✅ Model training completed$(NC)"

test: ## Run tests
	@echo "$(GREEN)🧪 Running tests...$(NC)"
	docker-compose run --rm api pytest -v
	@echo "$(GREEN)✅ Tests completed$(NC)"

dashboard-urls: ## Show all dashboard URLs
	@echo ""
	@echo "$(GREEN)📊 MLOps Dashboards & Services$(NC)"
	@echo "$(GREEN)================================$(NC)"
	@echo "$(BLUE)🌐 FastAPI (REST):$(NC)      http://localhost:8800"
	@echo "$(BLUE)📚 FastAPI Docs:$(NC)       http://localhost:8800/docs"
	@echo "$(BLUE)🧠 MLFlow:$(NC)             http://localhost:5000"
	@echo "$(BLUE)🔄 Airflow:$(NC)            http://localhost:8080 (admin/admin123)"
	@echo "$(BLUE)🗄️ MinIO:$(NC)              http://localhost:9001 (minioadmin/minioadmin123)"
	@echo "$(BLUE)🎯 GraphQL:$(NC)            http://localhost:4000/graphql"
	@echo "$(BLUE)📡 gRPC:$(NC)               localhost:50051"
	@echo "$(BLUE)🔥 Kafka UI:$(NC)           http://localhost:8088"
	@echo "$(BLUE)🖥️ Original UI:$(NC)        file://$(PWD)/ui/index.html"
	@echo ""
	@echo "$(YELLOW)🔧 Useful Commands:$(NC)"
	@echo "  make logs           - Show all logs"
	@echo "  make check-health   - Check service health"
	@echo "  make train-model    - Train ML model"
	@echo "  make test-apis      - Test all API endpoints"
	@echo ""

check-health: ## Check health of all services
	@echo "$(GREEN)🏥 Checking service health...$(NC)"
	@echo ""
	@echo "$(BLUE)FastAPI:$(NC)"
	@curl -s http://localhost:8800/health | jq '.' || echo "$(RED)❌ FastAPI not responding$(NC)"
	@echo ""
	@echo "$(BLUE)MLFlow:$(NC)"
	@curl -s http://localhost:5000/health | head -1 || echo "$(RED)❌ MLFlow not responding$(NC)"
	@echo ""
	@echo "$(BLUE)GraphQL:$(NC)"
	@curl -s http://localhost:4000/ | jq '.' || echo "$(RED)❌ GraphQL not responding$(NC)"
	@echo ""
	@echo "$(BLUE)MinIO:$(NC)"
	@curl -s http://localhost:9000/minio/health/live || echo "$(RED)❌ MinIO not responding$(NC)"
	@echo ""
	@echo "$(BLUE)Airflow:$(NC)"
	@curl -s -o /dev/null -w "%{http_code}" http://localhost:8080 | grep -q 200 && echo "$(GREEN)✅ Airflow OK$(NC)" || echo "$(RED)❌ Airflow not responding$(NC)"

test-apis: ## Test all API endpoints
	@echo "$(GREEN)🔧 Testing API endpoints...$(NC)"
	@echo ""
	@echo "$(BLUE)1. FastAPI Health:$(NC)"
	@curl -s http://localhost:8800/health | jq '.'
	@echo ""
	@echo "$(BLUE)2. Get OHLCV Data:$(NC)"
	@curl -s "http://localhost:8800/v1/crypto/ohlcv?symbol=BTCUSDT&limit=5" | jq '.data[:2]'
	@echo ""
	@echo "$(BLUE)3. Generate Heuristic Signal:$(NC)"
	@curl -s -X POST "http://localhost:8800/v1/crypto/signal" \
		-H "Content-Type: application/json" \
		-d '{"symbol":"BTCUSDT","explain":true}' | jq '.risk_score, .vol_regime'
	@echo ""
	@echo "$(BLUE)4. GraphQL Health:$(NC)"
	@curl -s -X POST "http://localhost:4000/graphql" \
		-H "Content-Type: application/json" \
		-d '{"query":"{ health { status mlAvailable } }"}' | jq '.data.health'
	@echo ""

generate-test-data: ## Generate test streaming data
	@echo "$(GREEN)📡 Generating test streaming data...$(NC)"
	docker-compose exec crypto-producer python -c "
	import time
	from crypto_producer import CryptoProducer
	producer = CryptoProducer()
	print('Sending 10 test messages...')
	for i in range(10):
	    for symbol in ['BTC/USDT', 'ETH/USDT']:
	        data = producer.simulate_price_movement(symbol)
	        producer.send_price_update(data)
	        print(f'Sent {symbol}: \$${data[\"price\"]:.2f}')
	    time.sleep(1)
	producer.close()
	print('Test data generation completed!')
	"

show-kafka-topics: ## Show Kafka topics and messages
	@echo "$(GREEN)📊 Kafka Topics Status:$(NC)"
	@docker-compose exec kafka kafka-topics --bootstrap-server localhost:9092 --list
	@echo ""
	@echo "$(GREEN)Recent crypto-prices messages:$(NC)"
	@timeout 5 docker-compose exec kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic crypto-prices --from-beginning --max-messages 5 || true
	@echo ""
	@echo "$(GREEN)Recent predictions messages:$(NC)"
	@timeout 5 docker-compose exec kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic predictions --from-beginning --max-messages 3 || true

backup-data: ## Backup important data
	@echo "$(GREEN)💾 Backing up data...$(NC)"
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	@cp -r data/ backups/$(shell date +%Y%m%d_%H%M%S)/
	@cp .env backups/$(shell date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
	@echo "$(GREEN)✅ Backup completed in backups/$(shell date +%Y%m%d_%H%M%S)$(NC)"

monitor: ## Show real-time monitoring
	@echo "$(GREEN)📈 Real-time monitoring (Press Ctrl+C to stop)$(NC)"
	@while true; do \
		clear; \
		echo "$(GREEN)=== CRYPTO MLOPS MONITORING ===$(NC)"; \
		echo "$(BLUE)Time:$(NC) $$(date)"; \
		echo ""; \
		echo "$(BLUE)Container Status:$(NC)"; \
		docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" | head -10; \
		echo ""; \
		echo "$(BLUE)Latest API Metrics:$(NC)"; \
		curl -s http://localhost:8800/metrics | jq '.' 2>/dev/null || echo "API not responding"; \
		echo ""; \
		echo "$(BLUE)Latest Signals (last 2):$(NC)"; \
		curl -s "http://localhost:8800/v1/crypto/signals/tail?n=2" | jq '.[].symbol, .[].risk_score, .[].vol_regime' 2>/dev/null | paste - - - || echo "No signals yet"; \
		echo ""; \
		echo "$(YELLOW)Refreshing in 10 seconds... (Ctrl+C to stop)$(NC)"; \
		sleep 10; \
	done

# Development shortcuts
dev-reset: clean setup ## Complete development reset
	@echo "$(GREEN)🔄 Development environment reset completed!$(NC)"

quick-start: setup-env up ## Quick start (without full setup)
	@echo "$(GREEN)⚡ Quick start completed! Services are starting...$(NC)"
	@sleep 10
	@make dashboard-urls