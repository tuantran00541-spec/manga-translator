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
	python -m compileall -q app tests
	pytest -q tests/test_auto_text_objects.py tests/test_translation_deepseek.py tests/test_chapter_export.py tests/test_downloader_discovery.py tests/test_v01_product_closure.py tests/test_phase41_data_model.py tests/test_phase42_review_artifacts.py tests/test_phase43_chapter_qc_ui.py tests/test_phase45_render_export.py tests/test_ui_system_contract.py

test-browser:
	python tests/browser_phase43_chapter_qc.py
	python tests/browser_phase44_ocr.py
	python tests/browser_phase45_render_export.py

release-check: test test-browser

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf data/raw/* data/processed/* data/output/* logs/*.log 2>/dev/null || true
