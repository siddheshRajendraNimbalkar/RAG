.PHONY: venv install run clean-db docker-start docker-stop docker-models

# Create Virtual Environment
venv:
	python -m venv venv

# Activate Environment (Windows)
activate:
	venv\Scripts\activate

# Install Requirements
install:
	pip install -r requirements.txt

# Run Ingestion Pipeline
ingest:
	python ingestion_pipeline.py

# Run Retrieval Pipeline
retrieve:
	python retrieval_pipeline.py

# Run History aware Generation Pipeline
history:
	python history_aware_generation.py

# Delete Chroma DB Folder
clean-db:
	rmdir /s /q db\chroma_db

# Start Ollama Docker Container
docker-start:
	docker run -d --name ollama -p 11434:11434 -v ollama:/root/.ollama ollama/ollama

# Stop Ollama Container
docker-stop:
	docker stop ollama

# Pull Required Models
docker-models:
	docker exec -it ollama ollama pull nomic-embed-text
	docker exec -it ollama ollama pull llama3.2:1b

# Show Installed Ollama Models
docker-list:
	docker exec -it ollama ollama list