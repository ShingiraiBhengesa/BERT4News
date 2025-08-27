.PHONY: install setup train index eval run test clean docker

# Environment setup
install:
	pip install -r requirements.txt
	python -m nltk.downloader punkt stopwords

setup: install
	mkdir -p data/raw data/processed data/artifacts
	mkdir -p logs
	python scripts/ingest.py

# Model training pipeline
train:
	python scripts/train_cf.py
	python scripts/build_tfidf.py

index:
	python scripts/build_embeddings.py
	python scripts/build_faiss_index.py

eval:
	python scripts/offline_eval.py

# Application
run:
	python api/app.py

run-prod:
	gunicorn --bind 0.0.0.0:5000 --workers 4 --worker-class gevent api.wsgi:app

# Development
test:
	pytest tests/ -v

lint:
	black . --check
	flake8 .
	mypy recsys/

format:
	black .

# Docker
docker-build:
	docker build -t bert4news .

docker-run:
	docker run -p 5000:5000 bert4news

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf data/artifacts/*

# Demo setup
demo:
	python scripts/seed_demo_users.py
	make train
	make index
	make run
