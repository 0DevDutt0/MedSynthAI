.PHONY: setup

setup:
	@echo "Creating virtual environment..."
	@python3 -m venv .venv
	@source .venv/bin/activate && pip install --upgrade pip
	@source .venv/bin/activate && pip install -r requirements.txt
	@echo "Creating project directories..."
	@mkdir -p data logs reports models src tests alerts scripts
	@echo "Setup complete!"