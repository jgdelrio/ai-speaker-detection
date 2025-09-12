# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


## Development Commands

This project uses `uv` as the package manager.

### Installation and Setup
```bash
uv sync --dev  # Install dependencies including dev dependencies
```

### Code Quality
```bash
black .  # Format code (line length: 88 characters)
flake8 .  # Lint code
```

### Testing
```bash
pytest  # Run all tests
pytest tests/specific_test.py  # Run specific test file
pytest -v  # Verbose test output
pytest -k "test_name"  # Run specific test by name
```

### Running the Service
```bash
python -m app.main  # Run the FastAPI server (localhost:8000)
uvicorn app.main:app --reload  # Run with auto-reload for development
```

## Architecture Overview

This is a FastAPI-based speaker identification service that processes audio files and transcription data to identify different speakers.

### Core Components

**FastAPI Application** (`app/main.py`):
- Single FastAPI instance with two router modules
- Service runs on port 8000 by default

**API Routes** (`app/api/routes/`):
- `/speaker-identification` - Main endpoint that accepts audio file + transcription data
- Healthcheck endpoint for service monitoring

**Speaker Processing**:
- `SpeakerService` (`app/services/speaker_service.py`) - Main business logic service (currently returns mock data)
- `identification.py` contains SpeechBrain ECAPA-VOXCELEB model integration for actual speaker embedding extraction
- Model downloads to `tmp_model/` directory and runs on CPU

**Data Models** (`app/models/`):
- Pydantic models for request/response validation
- `SpeakerIdentificationResponse` wraps enhanced transcription data with speaker analysis

### Key Dependencies
- **FastAPI** - Web framework
- **SpeechBrain** - Speaker recognition model (ECAPA-VOXCELEB)
- **torchaudio** - Audio processing
- **librosa** - Audio analysis utilities
- **scikit-learn** - ML utilities

### Configuration
- Settings managed via `pydantic-settings` with `.env` file support
- Service metadata defined in `app/core/config.py`

## Current Implementation Status

The service currently returns mock speaker analysis data. The actual SpeechBrain model integration exists in `app/services/identification.py` but is not yet connected to the main service pipeline.

## Project Overview

This is an AI speaker detection service that identifies different speakers in audio files and updates transcription information. The project is built as a FastAPI web service using Python 3.12.

## Architecture

The project follows a typical FastAPI structure:

- `run_detect.py` - Main entry point that starts the uvicorn server
- `app/` - Main application package
  - `core/` - Core utilities and configuration
    - `warnings_filter.py` - Suppresses known deprecation warnings from dependencies (PyTorch, SpeechBrain, etc.)
  - `api/` - API routes and endpoints (currently empty)
  - `services/` - Business logic and services (currently empty)

## Key Dependencies

Based on the warnings filter, this project uses:
- PyTorch for deep learning
- SpeechBrain for speech processing 
- torchaudio for audio processing
- FastAPI/uvicorn for the web service
- Various AWS and data processing libraries

## Development Setup

The project uses uv as the package manager and is configured with `pyproject.toml`:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the development server
uv run python run_detect.py

# Or run the Lambda handler locally
uv run python lambda_main.py
```

## AWS Lambda Deployment

The project includes AWS Lambda deployment support:

- `Dockerfile.lambda` - Docker image for AWS Lambda container deployment
- `lambda_main.py` - Lambda handler with FastAPI app and Mangum integration

### Lambda Deployment Commands

```bash
# Build Lambda Docker image
docker build -f Dockerfile.lambda -t ai-speaker-detection-lambda .

# Tag for ECR (replace with your registry)
docker tag ai-speaker-detection-lambda:latest <account-id>.dkr.ecr.<region>.amazonaws.com/ai-speaker-detection:latest

# Push to ECR
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/ai-speaker-detection:latest
```

### Lambda API Endpoints

- `GET /` or `GET /health` - Health check endpoint
- `POST /detect-speakers` - Main speaker detection endpoint

Expected request format:
```json
{
  "audio_file": "path/to/audio.mp3",
  "transcription": "text to update with speaker info",
  "expected_speakers": 2
}
```

## Missing Files

The project was incomplete but now includes Lambda deployment. Originally missing:
- `app/main.py` - Now implemented in `lambda_main.py`
- `app/core/config.py` - Configuration handled in Lambda handler

## Test Data

- `speaker_detection_test_results.json` - Contains test results showing speaker detection performance on WhatsApp audio files
- `media/` - Likely contains test audio files
- `tmp_model/` - Temporary model storage directory

## Docker Image management

Commands to login into ECR, build and push the image
- ECR login: `aws ecr get-login-password --region eu-west-1 --profile synth | docker login --username AWS --password-stdin 575380175069.dkr.ecr.eu-west-1.amazonaws.com`
- Build image: `docker build -t ai-scriber-transcriptor:<version> .`
- Tag image: `docker tag ai-scriber-transcriptor:<version> 575380175069.dkr.ecr.eu-west-1.amazonaws.com/scriber:speaker-transcriptor-<version>`
- Push image: `docker push 575380175069.dkr.ecr.eu-west-1.amazonaws.com/scriber:speaker-transcriptor-<version>`

## Notes

- The project suppresses various deprecation warnings from ML dependencies
- Test results show single-speaker detection with 95% confidence
- Processing times range from 1-4 seconds for various audio lengths
