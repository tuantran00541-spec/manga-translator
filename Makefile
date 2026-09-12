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
	python -m pytest -q

test-browser:
	python scripts/browser_sanity.py
	node scripts/editor_auto_sync_sanity.js
	node scripts/studio_runtime_sanity.js
	node scripts/theme_runtime_sanity.js
	@echo "Run scripts/live_ui_smoke.py against a started server for real Chromium coverage."

release-check: test test-browser
	python scripts/release_sanity.py
	python scripts/process_responsiveness_sanity.py
	python scripts/inpaint_authority_sanity.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache playwright-report test-results htmlcov coverage.xml .coverage 2>/dev/null || true
	rm -rf data/raw/* data/processed/* data/output/* logs/*.log 2>/dev/null || true
