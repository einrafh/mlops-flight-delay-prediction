# Use a lightweight official Python image
FROM python:3.10-slim

# Set the working directory within the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install core dependencies and API frameworks
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn pydantic

# Copy the source code into the container
COPY src/ ./src/

# Expose the port that the API will run on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]