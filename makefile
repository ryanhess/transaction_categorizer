.PHONY: clean

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.ruff_cache" -exec rm -rf {} +

train:
	python -c "from transaction_categorizer.inference.cat.train import train; print(train())"