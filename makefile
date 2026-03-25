.PHONY: run

server-start:
	uvicorn src.main:app --reload --port 8001