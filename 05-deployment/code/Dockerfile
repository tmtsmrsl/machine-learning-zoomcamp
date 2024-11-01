FROM python:3.12.7-slim

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

# Copy application files
COPY ["predict.py", "model_C=1.0.bin", "./"]

# Expose the application's port
EXPOSE 9696

# Define the entrypoint for the application
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]