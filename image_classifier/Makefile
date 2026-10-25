#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = image_classifier
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	$(PYTHON_INTERPRETER) -c "from pathlib import Path; import shutil; roots=(Path('image_classifier'), Path('tests')); [path.unlink() for root in roots for pattern in ('*.pyc', '*.pyo') for path in root.rglob(pattern)]; [shutil.rmtree(path) for root in roots for path in root.rglob('__pycache__') if path.is_dir()]"


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run ruff format --check
	uv run ruff check

## Format source code with ruff
.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format



## Run tests
.PHONY: test
test:
	uv run python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Download and verify CIFAR-10
.PHONY: data
data:
	uv run python -m image_classifier.dataset


## Train, evaluate, and save the CIFAR-10 classifier
.PHONY: train
train:
	uv run python -m image_classifier.main train


## Classify a sample from the CIFAR-10 test set
.PHONY: predict
predict:
	uv run python -m image_classifier.main predict-sample


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
