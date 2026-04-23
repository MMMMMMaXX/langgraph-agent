PYTHON ?= ./.venv/bin/python
BASE_URL ?= http://127.0.0.1:8000
EVAL_PROFILE ?= baseline
EVAL_CASE_IDS ?=
PROD_COMPOSE ?= docker-compose.prod.yml
BACKUP_DIR ?= backups
BACKUP_PATH ?=
VOLUME_NAME ?= langgraph_data

.PHONY: docker-up docker-down docker-restart docker-logs docker-health docker-smoke prod-up prod-down prod-restart prod-logs prod-health prod-smoke prod-backup prod-restore eval-baseline

docker-up:
	docker compose up --build

docker-down:
	docker compose down

docker-restart:
	docker compose restart

docker-logs:
	docker compose logs -f langgraph-agent

docker-health:
	curl -f $(BASE_URL)/health

docker-smoke:
	BASE_URL=$(BASE_URL) ./scripts/smoke_docker.sh

prod-up:
	docker compose -f $(PROD_COMPOSE) up --build -d

prod-down:
	docker compose -f $(PROD_COMPOSE) down

prod-restart:
	docker compose -f $(PROD_COMPOSE) restart

prod-logs:
	docker compose -f $(PROD_COMPOSE) logs -f langgraph-agent

prod-health:
	curl -f $(BASE_URL)/health

prod-smoke:
	BASE_URL=$(BASE_URL) COMPOSE_FILE=$(PROD_COMPOSE) ./scripts/smoke_docker.sh

prod-backup:
	BACKUP_DIR=$(BACKUP_DIR) VOLUME_NAME=$(VOLUME_NAME) ./scripts/backup_docker_volume.sh

prod-restore:
	BACKUP_PATH=$(BACKUP_PATH) VOLUME_NAME=$(VOLUME_NAME) CONFIRM_RESTORE=$(CONFIRM_RESTORE) ./scripts/restore_docker_volume.sh

eval-baseline:
	EVAL_BASE_URL=$(BASE_URL) EVAL_CASE_IDS="$(EVAL_CASE_IDS)" $(PYTHON) scripts/run_eval_profile.py --profile $(EVAL_PROFILE)
