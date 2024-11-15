include .env

build:
	uv build

publish:
	uv publish --token ${UV_PUBLISH_TOKEN}

typecheck:
	uvx mypy --python-executable "./.venv/bin/python3.10" .
	uvx pyright --pythonpath "./.venv/bin/python3.10"
