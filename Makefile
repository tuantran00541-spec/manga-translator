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

test:
	python -m compileall -q app run.py scripts

test-browser:
	@echo "Browser regression tests are archived on archive/main-tests-20260828"

release-check: test
	python scripts/release_sanity.py
	python scripts/inpaint_authority_sanity.py
	python scripts/browser_sanity.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache playwright-report test-results htmlcov coverage.xml .coverage 2>/dev/null || true
	rm -rf data/raw/* data/processed/* data/output/* logs/*.log 2>/dev/null || true
