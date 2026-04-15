# Deploy targets for the wildlife webapp.
#
# Usage:
#   make deploy        — sync local repo + restart the remote stack
#   make sync          — rsync only (no Docker action)
#   make remote-up     — bring the remote stack up (build + up -d)
#   make remote-down   — stop the remote stack
#   make remote-logs   — tail the webapp container logs
#
# Override the host or path on the command line, e.g.:
#   make deploy REMOTE_HOST=ada@server REMOTE_PATH=/srv/foo

REMOTE_HOST   ?= cwinkelmann@10.188.1.1
REMOTE_PATH   ?= /raid/cwinkelmann/work/usde-innovations-applications-forest-it
COMPOSE_FILES := -f docker-compose.yml -f docker-compose.gpu.yml

# Excluded paths — heavy / transient / machine-specific. .env is host-
# specific (NUM_WORKERS, DEVICE, BATCH_SIZE_*, etc.), so we never push it
# from the laptop to the server. Keep separate `.env` files on each side.
RSYNC_EXCLUDES := \
	--exclude='.git/' \
	--exclude='__pycache__/' \
	--exclude='*.pyc' \
	--exclude='.pytest_cache/' \
	--exclude='.ipynb_checkpoints/' \
	--exclude='.DS_Store' \
	--exclude='.venv/' \
	--exclude='.idea/' \
	--exclude='.vscode/' \
	--exclude='.claude/' \
	--exclude='node_modules/' \
	--exclude='wandb/' \
	--exclude='output/' \
	--exclude='outputs/' \
	--exclude='weights/' \
	--exclude='webapp/_data/' \
	--exclude='_data/' \
	--exclude='.playwright-mcp/' \
	--exclude='.env.local'

# rsync flags:
#   -a  archive (preserves perms / times / symlinks)
#   -v  verbose so you see what moved
#   -z  compress on the wire
#   -h  human-readable sizes
#   --info=progress2  single-line overall progress (rsync >= 3.1)
RSYNC_FLAGS := -avzh --info=progress2

.PHONY: deploy sync remote-build remote-up remote-down remote-logs remote-shell

deploy: sync remote-down remote-up
	@echo ">>> deploy complete — $(REMOTE_HOST):$(REMOTE_PATH)"

sync:
	@echo ">>> rsync to $(REMOTE_HOST):$(REMOTE_PATH)"
	rsync $(RSYNC_FLAGS) $(RSYNC_EXCLUDES) ./ $(REMOTE_HOST):$(REMOTE_PATH)/

remote-down:
	@echo ">>> docker compose down on $(REMOTE_HOST)"
	ssh $(REMOTE_HOST) 'cd $(REMOTE_PATH)/webapp && docker compose $(COMPOSE_FILES) down'

remote-build:
	@echo ">>> docker compose build on $(REMOTE_HOST)"
	ssh $(REMOTE_HOST) 'cd $(REMOTE_PATH)/webapp && docker compose $(COMPOSE_FILES) build'

remote-up:
	@echo ">>> docker compose up -d --build on $(REMOTE_HOST)"
	ssh $(REMOTE_HOST) 'cd $(REMOTE_PATH)/webapp && docker compose $(COMPOSE_FILES) up -d --build'

remote-logs:
	ssh $(REMOTE_HOST) 'cd $(REMOTE_PATH)/webapp && docker compose $(COMPOSE_FILES) logs -f --tail=200 webapp'

remote-shell:
	ssh -t $(REMOTE_HOST) 'cd $(REMOTE_PATH)/webapp && docker compose $(COMPOSE_FILES) exec webapp /usr/local/bin/_entrypoint.sh bash'

# ── Docker housekeeping ────────────────────────────────────────────────────
# Two tiers:
#   *-prune      — safe: stopped containers, dangling <none> images,
#                  build cache, unused networks. Tagged images stay.
#   *-prune-all  — aggressive: also drops unused TAGGED images + volumes.
#                  On a shared server, this affects images other users
#                  may rely on. Run deliberately.

.PHONY: docker-prune docker-prune-all remote-docker-prune remote-docker-prune-all

docker-prune:
	@echo ">>> local: prune stopped containers, dangling images, build cache, networks"
	docker container prune -f
	docker image prune -f
	docker builder prune -f
	docker network prune -f
	docker system df

docker-prune-all:
	@echo ">>> local: AGGRESSIVE prune (unused tagged images + volumes too)"
	docker system prune -a -f --volumes
	docker builder prune -a -f
	docker system df

remote-docker-prune:
	@echo ">>> $(REMOTE_HOST): prune stopped containers, dangling images, build cache, networks"
	ssh $(REMOTE_HOST) ' \
	  docker container prune -f && \
	  docker image prune -f && \
	  docker builder prune -f && \
	  docker network prune -f && \
	  docker system df'

remote-docker-prune-all:
	@echo ">>> $(REMOTE_HOST): AGGRESSIVE prune (unused tagged images + volumes too)"
	ssh $(REMOTE_HOST) ' \
	  docker system prune -a -f --volumes && \
	  docker builder prune -a -f && \
	  docker system df'
