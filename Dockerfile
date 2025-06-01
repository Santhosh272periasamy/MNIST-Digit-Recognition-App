FROM python:3.11.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files to the container
COPY . .

# Expose port 8080 (required for GCP Cloud Run)
EXPOSE 8080

# Start Streamlit on port 8080 and make it accessible externally
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
