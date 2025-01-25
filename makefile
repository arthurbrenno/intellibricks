.PHONY: run release test docs clean typecheck

.DEFAULT_GOAL := run

run:
	./.venv/bin/python ./tests/example.py

# Creates a new release in the GitHub repository.
release:
	uv run python scripts/release.py

test:
	uv run pytest -x -s

docs:
	@echo "Gerando documentação..."
	cd docs && make html

clean:
	@echo "Removendo diretórios __pycache__..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Limpeza concluída."

# Runs typechecks using mypy and pyright.
typecheck:
	uvx pyright --pythonpath "./.venv/bin/python3.10" src
	uvx mypy --python-executable "./.venv/bin/python3.10" src
