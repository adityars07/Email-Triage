FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 7860

# Set environment variables for Hugging Face Spaces
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

# Run the FastAPI + Gradio app
CMD ["python", "app.py"]
