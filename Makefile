PYTHON ?= python3
NODE ?= node

.PHONY: help check check-architecture check-runtime check-research check-web analysis-api jury-api review-ui

help:
	@echo 'check: CPU regression tests, no training or GPU inference'
	@echo 'check-web: browser/Next proxy tests (frontend dependencies required)'
	@echo 'analysis-api / jury-api / review-ui: explicit service entrypoints'

check: check-architecture check-runtime check-research

check-architecture:
	$(PYTHON) -m unittest discover -s tests -t . -v

check-runtime:
	PYTHONPATH=ml $(PYTHON) -m unittest ml.gemma_lab.test_protocol ml.gemma_lab.test_bridge ml.gemma_lab.test_regulatory ml.gemma_lab.test_runner_instructions ml.gemma_lab.test_detail_selection ml.context_pipeline.test_verdict ml.narration_training.organizer.test_continuity ml.test_jury_api ml.test_review_frames ml.test_review_frames_routes

check-research:
	PYTHONPATH=ml $(PYTHON) -m unittest discover -s ml/gemma_lab/adaptation -t . -p 'test_*.py'

check-web:
	$(NODE) ml/gemma_lab/test_ui.cjs
	cd frontend && $(NODE) --import tsx --test tests/gemma-proxy.test.ts

analysis-api:
	$(PYTHON) -m forte_vision.interfaces.analysis_http --port 8771

jury-api:
	NUHACK_JURY_ENGINE=gemma $(PYTHON) -m forte_vision.interfaces.jury_http --port 8765

review-ui:
	$(PYTHON) -m forte_vision.interfaces.review_http --port 8765
