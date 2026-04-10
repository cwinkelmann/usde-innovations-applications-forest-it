UID := $(shell id -u)

.PHONY: build up shell logs clean help

build: ## Build the fit-training Docker image
	UID=$(UID) docker compose build

up: ## Start JupyterLab (http://localhost:8888)
	UID=$(UID) docker compose up

shell: ## Open a bash shell inside the container
	UID=$(UID) docker compose run --service-ports \
	    --entrypoint /bin/bash fit-training

logs: ## Follow container logs
	docker compose logs -f fit-training

clean: ## Remove stopped containers and the week1_data volume
	docker compose down --volumes

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	    | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'


