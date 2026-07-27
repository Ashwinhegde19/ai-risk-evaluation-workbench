# Makefile — AI Risk Evaluation Workbench
# Convenience targets for install, test, eval, and dashboard workflows.

.PHONY: help install install-dev test coverage eval-mock demo dashboard clean

.DEFAULT_GOAL := help

help: ## List available targets
	@echo "AI Risk Evaluation Workbench — make targets"
	@echo ""
	@echo "  install      Install the package (pip install -e .)"
	@echo "  install-dev  Install with dev extras (pip install -e \".[dev]\")"
	@echo "  test         Run the test suite (pytest)"
	@echo "  coverage     Run tests under coverage and print a report"
	@echo "  eval-mock    Run the eval pipeline in mock mode"
	@echo "  demo         Generate demo artifacts (python -m src.demo)"
	@echo "  dashboard    Launch the Streamlit dashboard"
	@echo "  clean        Remove caches, coverage output, and build artifacts"

install: ## Install the package in editable mode
	pip install -e .

install-dev: ## Install the package with dev extras
	pip install -e ".[dev]"

test: ## Run the full test suite
	python3 -m pytest tests/ -v

coverage: ## Run tests under coverage and report
	python3 -m coverage run --source=src -m pytest tests/ && python3 -m coverage report

eval-mock: ## Run the eval pipeline in mock mode
	python3 -m src.pipeline.run --model gpt-4o --mock --report-dir results

demo: ## Generate demo artifacts
	python3 -m src.demo

dashboard: ## Launch the Streamlit dashboard
	streamlit run src/dashboard/app.py

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov *.egg-info
