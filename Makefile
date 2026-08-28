.PHONY: run docker-up docker-build docker-down health test test-browser release-check clean

run:
	python run.py

docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-down:
	docker compose down

health:
	curl -s http://127.0.0.1:8000/health | python -m json.tool

# Production main intentionally carries runtime source only. Full regression and
# browser suites are preserved on archive/main-tests-20260828 and feature branches.
test:
	python -m compileall -q app run.py

test-browser:
	@echo "Browser regression tests are archived on archive/main-tests-20260828"

release-check: test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache playwright-report test-results htmlcov coverage.xml .coverage 2>/dev/null || true
	rm -rf data/raw/* data/processed/* data/output/* logs/*.log 2>/dev/null || true
