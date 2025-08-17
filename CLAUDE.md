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