# Makefile for AI Inference Energy project
# Provides convenient commands for development, testing, and deployment

.PHONY: help install install-dev test lint format clean docs gpu-test enterprise-test

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run unit tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting and type checking"
	@echo "  format       - Format code with black and isort"
	@echo "  format-check - Check code formatting without changes"
	@echo "  clean        - Clean build artifacts and cache"
	@echo "  docs         - Build documentation"
	@echo "  gpu-test     - Test GPU hardware validation"
	@echo "  enterprise-test - Test Enterprise Linux compatibility"
	@echo "  security     - Run security checks"
	@echo "  pre-commit   - Install pre-commit hooks"
	@echo "  validate     - Run full validation suite"

# Installation targets
install:
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt

# Testing targets
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing

test-integration:
	@echo "Testing hardware module..."
	python -c "from hardware.gpu_info import get_gpu_info; print('✓ Hardware module OK')"
	@echo "Testing configuration consistency..."
	python -c "from config import GPUConfig; print('✓ Configuration module OK')"
	@echo "Testing sample scripts..."
	cd sample-collection-scripts && bash -n launch.sh && echo "✓ Launch script syntax OK"

# Code quality targets
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	mypy . --ignore-missing-imports --no-strict-optional || true

format:
	black --line-length=127 .
	isort --profile=black --line-length=127 .

format-check:
	black --check --line-length=127 .
	isort --check-only --profile=black --line-length=127 .

# Security targets
security:
	bandit -r . -f json -o bandit-report.json --exit-zero
	safety check -r requirements.txt --continue-on-error

# GPU-specific testing
gpu-test:
	@echo "Testing GPU frequency configurations..."
	python -c "from hardware.gpu_info import get_gpu_info, get_supported_gpus; \
		[print(f'✓ {gpu}: {len(info.get_available_frequencies())} frequencies ({min(info.get_available_frequencies())}-{max(info.get_available_frequencies())} MHz)') \
		for gpu in get_supported_gpus() \
		for info in [get_gpu_info(gpu)] \
		if all(f >= 510 for f in info.get_available_frequencies()) or exit(f'{gpu} has frequencies below 510 MHz')]; \
		print('All GPU configurations validated')"

# Enterprise Linux testing (requires Docker)
enterprise-test:
	@echo "Testing Rocky Linux 9 compatibility..."
	docker run --rm -v $(PWD):/workspace -w /workspace rockylinux:9 /bin/bash -c "\
		dnf update -y && \
		dnf install -y python3 python3-pip python3-devel gcc && \
		python3 -m pip install --upgrade pip && \
		python3 -m pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu && \
		python3 -m pip install -r requirements.txt && \
		python3 -c 'from hardware.gpu_info import get_supported_gpus; print(\"✓ Rocky Linux compatibility OK\")'"

# Documentation targets
docs:
	@echo "Validating documentation files..."
	python -c "\
	import os; \
	for root, dirs, files in os.walk('.'): \
		for file in files: \
			if file.endswith('.md'): \
				filepath = os.path.join(root, file); \
				try: \
					with open(filepath, 'r', encoding='utf-8') as f: \
						content = f.read(); \
					print(f'✓ {filepath}'); \
				except Exception as e: \
					print(f'✗ {filepath}: {e}'); \
					exit(1)"

# Pre-commit setup
pre-commit:
	pre-commit install
	@echo "Pre-commit hooks installed. Run 'pre-commit run --all-files' to test."

# Cleanup targets
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.log" -delete
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	rm -f bandit-report.json coverage.xml

# Full validation pipeline (mimics CI)
validate: format-check lint test-integration gpu-test security
	@echo "✅ All validation checks passed!"

# Development setup (run once)
setup: install-dev pre-commit
	@echo "Development environment setup complete!"
	@echo "Run 'make validate' to test your setup."
