# Creates a new release in the GitHub repository.
release:
	uv run python scripts/release.py

# Runs typechecks using mypy and pyright.
typecheck:
	uvx mypy --python-executable "./.venv/bin/python3.10" src
	uvx pyright --pythonpath "./.venv/bin/python3.10" src
