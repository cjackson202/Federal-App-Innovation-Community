FROM python:3.11-slim

WORKDIR /app

# Install uv for faster package management
RUN pip install uv

# Copy requirements file
COPY requirements.txt .

# Install dependencies using uv
RUN uv venv
RUN uv pip install -r requirements.txt

# Copy application code
COPY server_sse.py .

# Expose the port the server runs on
EXPOSE 8080

# Command to run the server
CMD ["uv", "run", "server_sse.py"] 