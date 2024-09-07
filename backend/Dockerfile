# Use the official Python image as the base image
FROM python:3.12-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the environment variable to use a specific directory in Docker
ENV ENV_FOR_DYNACONF=production

# Install Poetry
RUN apt-get update && apt-get install -y curl \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get clean

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Copy the poetry.lock and pyproject.toml files first to leverage Docker cache
# COPY README.md pyproject.toml poetry.lock ./
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-root --no-interaction --no-ansi --no-dev

# Copy the rest of the application code to the container
COPY . .

# Install project
RUN poetry install --only-root --no-interaction --no-ansi --no-dev

# Expose the port that the FastAPI app will run on
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["poetry", "run", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]