FROM python:3.13-slim
WORKDIR /app

# Install uv globally
RUN pip install uv

# Copy requirements and install GLOBALLY to the system using --system
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

# Default command (Docker Compose can override this to 'fastapi dev')
CMD ["fastapi", "run", "server.py", "--port", "8000"]