.PHONY: test test-fast install

install:
	pip install -r requirements.txt --break-system-packages
	pip install pytest-cov httpx --break-system-packages

# Full run: coverage report + fails the build under 75% coverage (see pytest.ini)
test:
	python3 -m pytest

# Quick run for local iteration: no coverage overhead
test-fast:
	python3 -m pytest --no-cov -q
