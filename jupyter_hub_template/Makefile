# folder to create Python virtual environment in
VENV ?= .venv
# setting PATH to use tools from the virtual environment
PATH := $(VENV)/bin:$(PATH)

JUPYTER := $(VENV)/bin/jupyter
JUPYTER_IP ?= 127.0.0.1
.PHONY: jupyterlab
jupyterlab: $(JUPYTER) ## Start JupyterLab
	$(JUPYTER) lab --ip=$(JUPYTER_IP)


#.PHONY: shell
#shell: $(PYTHON) ## Activate virtual environment
#	bash -c ". $(VENV)/bin/activate"


.PHONY: install
install: $(JUPYTER) ## Set up .venv and install dependencies
	# disable "Would you like to get notified about official Jupyter news?" pop-up
	# see https://jupyterlab.readthedocs.io/en/stable/user/announcements.html#configuration
	$(JUPYTER) labextension disable "@jupyterlab/apputils-extension:announcements"

PIP := $(VENV)/bin/pip
PYTHON := $(VENV)/bin/python3
$(JUPYTER): $(PYTHON) requirements.txt
# TODO
# check for Python version
# (.venv) $ python3 -c 'import platform; v = platform.python_version(); e = "3.11.3"; assert v == e, f"Found Python version {v} but expected {e}.";'
	$(PIP) install wheel
	$(PIP) install -r requirements.txt
	# TODO reruns even though it shouldn't
	@touch requirements.txt
	@touch $(JUPYTER)

# Python version to be used
PYTHON_VERSION ?= 3.11
$(PYTHON):
	python$(PYTHON_VERSION) -m venv $(VENV)
	-@touch $(PYTHON)


# recursive wildcard
rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2)$(filter $(subst *,%,$2),$d))
notebook_files=$(call rwildcard,,*.ipynb)


# execute all notebooks
.PHONY: run-notebooks
# export first because any notebook may use exported code
run-notebooks: export-notebooks $(notebook_files) ## Execute all notebooks
	$(JUPYTER) run jupyter nbconvert --to notebook --execute --stdout $(notebook_files)


# save notebooks with outputs to HTML
.PHONY: html-notebooks
html-notebooks: export-notebooks $(addsuffix .html,$(basename $(addprefix html/,$(notebook_files:notebooks/%=%)))) ## Save notebooks with outputs as HTML

# generate HTML file for notebook
html/%.html: notebooks/%.ipynb
	@echo "Converting notebook $< to HTML file $@"
	@mkdir -p "$(dir $@)"
	$(JUPYTER) nbconvert --to html --execute "$<" --stdout > "$@"


# build and verify docker image for working on mybinder.org
# https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html#preparing-your-dockerfile
.PHONY: test-image
test-image: ## Test docker image
	curl --location http://localhost:8888/ > /dev/null; test $$? -ne 0
	docker build -t binder-image .
	test $$(docker run --rm binder-image id -u) = 1000
	docker run --rm binder-image make style-notebooks
	docker run --rm binder-image make run-notebooks
	container_id=$$(docker run --detach --rm -p 8888:8888 binder-image jupyter-lab --ip=0.0.0.0 --port=8888); \
	trap 'docker kill $$container_id' EXIT; \
	sleep 2; \
	curl --location http://localhost:8888/ > /dev/null;


# check style for all notebooks
.PHONY: style-notebooks
style-notebooks: $(addsuffix .style,$(notebook_files)) ## Verify coding style in notebooks

.PHONY: %.ipynb.style
%.ipynb.style: %.ipynb
	cat "$<" | scripts/style.sh "$<"


.PHONY: export-notebooks
export-notebooks: $(addsuffix .py,$(basename $(addprefix project/,$(notebook_files:notebooks/%=%)))) ## Export code from notebooks

# generate python file from exported cells for notebook
project/%.py: notebooks/%.ipynb
	@echo "Exporting code cells from $< to $@"
	@mkdir -p "$(dir $@)"
	$(JUPYTER) nbconvert --to python "$<" --stdout --template export > "$@"


# target to sync notebook files with Python text files for using with an editor
# or integrated development environment
.PHONY: sync
sync: ## Sync notebook files to Python files and back
	scripts/sync-notebooks.sh $(notebook_files)


# docker setup using docker compose
# note this blends in the local files (see docker-compose.yml)
# this way files on the local host can be changed within docker
.PHONY: docker
docker: Makefile Dockerfile docker-compose.yml ## Build and run Docker image
	docker compose build --build-arg user_id=$(shell id -u)
	docker compose run --service-ports --entrypoint "/bin/bash" jupyterlab


# TODO
# look up in GNUMake book
.PHONY: clean
clean: ## Clean the project


.PHONY: distclean
distclean: clean ## Distclean the project
	$(RM)r $(VENV)/ sync/


.PHONY: help
help: ## Display this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
