#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = face_recognition
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
	$(PYTHON_INTERPRETER) -c "from pathlib import Path; import shutil; roots=(Path('face_recognition'), Path('tests')); [path.unlink() for root in roots for pattern in ('*.pyc', '*.pyo') for path in root.rglob(pattern)]; [shutil.rmtree(path) for root in roots for path in root.rglob('__pycache__') if path.is_dir()]"


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


## Inspect local gallery and query images
.PHONY: data
data:
	uv run python -m face_recognition.dataset


## Recognize one image using the local gallery
.PHONY: recognize
recognize:
	uv run python -m face_recognition.main recognize --image "$(IMAGE)"


## Recognize all images in data/test and save a CSV report
.PHONY: recognize-test-set
recognize-test-set:
	uv run python -m face_recognition.main recognize-test-set


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
