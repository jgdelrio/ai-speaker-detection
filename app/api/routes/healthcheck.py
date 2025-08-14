from fastapi import APIRouter

from app.core.config import SERVICE_VERSION

router = APIRouter()


@router.get("/healthcheck")
async def healthcheck():
    return {"status": "ok", "status_code": 200, "version": SERVICE_VERSION}
