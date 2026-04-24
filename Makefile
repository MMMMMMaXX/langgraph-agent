PYTHON ?= ./.venv/bin/python
BASE_URL ?= http://127.0.0.1:8000
EVAL_PROFILE ?= baseline
EVAL_CASE_IDS ?=
PROD_COMPOSE ?= docker-compose.prod.yml
BACKUP_DIR ?= backups
BACKUP_PATH ?=
VOLUME_NAME ?= langgraph_data

# ---- Version tagging ----
# 所有 docker build / up 都会把这三个值注入镜像，并通过 /health 暴露。
# 可覆盖：make build APP_VERSION=v0.3.1 IMAGE_TAG=v0.3.1
APP_VERSION   ?= dev
APP_GIT_SHA   ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
APP_BUILD_TIME ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
# 默认镜像 tag = 版本-短 sha，IMAGE_TAG=local/prod 时回退到固定 tag
IMAGE_TAG     ?= $(APP_VERSION)-$(APP_GIT_SHA)

# 把版本相关变量透传给 docker compose（compose 文件用 ${IMAGE_TAG:-...} 之类读取）
COMPOSE_ENV = APP_VERSION=$(APP_VERSION) APP_GIT_SHA=$(APP_GIT_SHA) APP_BUILD_TIME=$(APP_BUILD_TIME) IMAGE_TAG=$(IMAGE_TAG)

.PHONY: version build docker-up docker-down docker-restart docker-logs docker-health docker-smoke prod-up prod-down prod-restart prod-logs prod-health prod-smoke prod-backup prod-restore eval-baseline

version:
	@echo "APP_VERSION=$(APP_VERSION)"
	@echo "APP_GIT_SHA=$(APP_GIT_SHA)"
	@echo "APP_BUILD_TIME=$(APP_BUILD_TIME)"
	@echo "IMAGE_TAG=$(IMAGE_TAG)"

build:
	$(COMPOSE_ENV) docker compose build
	@echo "built image: langgraph-agent:$(IMAGE_TAG)"

docker-up:
	$(COMPOSE_ENV) docker compose up --build

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
	$(COMPOSE_ENV) docker compose -f $(PROD_COMPOSE) up --build -d

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
