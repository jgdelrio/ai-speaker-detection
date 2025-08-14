from fastapi import FastAPI
from app.api.routes import healthcheck, speaker_identification
from app.core.config import SERVICE_NAME, SERVICE_VERSION

app = FastAPI(
    title=SERVICE_NAME,
    description="Service for speaker identification",
    version=SERVICE_VERSION,
)

app.include_router(healthcheck.router)
app.include_router(speaker_identification.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
