import uvicorn
from app.main import app
from app.core import config as cfg


def debug_app():
    uvicorn.run(
        app, host=cfg.SERVICE_HOST, port=cfg.SERVICE_PORT, log_level="info"
    )


if __name__ == '__main__':
    debug_app()
