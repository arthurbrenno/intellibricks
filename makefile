include .env

build:
	uv build

publish:
	uv publish --token ${UV_PUBLISH_TOKEN}

typecheck:
	uvx mypy .
	uvx pyright
