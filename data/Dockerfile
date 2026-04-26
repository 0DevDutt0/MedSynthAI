# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY logs/ ./logs/
COPY alerts/ ./alerts/
COPY reports/ ./reports/

# Create necessary directories
RUN mkdir -p logs alerts reports

# Expose port for the LitServe server (if needed)
EXPOSE 8000

# Command to run the application
CMD ["python", "src/stream_simulator.py"]