.PHONY: help test clean typecheck

# Creates a new release in the GitHub repository.
release:
	uv run python scripts/release.py

test:
	uv run pytest -x -s

clean:
	@echo "Removendo diretórios __pycache__..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Limpeza concluída."

# Runs typechecks using mypy and pyright.
typecheck:
	uvx pyright --pythonpath "./.venv/bin/python3.10" src
	uvx mypy --python-executable "./.venv/bin/python3.10" src
