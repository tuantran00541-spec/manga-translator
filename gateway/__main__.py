import os

import uvicorn

from gateway.app import app_from_env

if __name__ == "__main__":
    uvicorn.run(app_from_env(), host=os.getenv("GATEWAY_HOST", "127.0.0.1"), port=int(os.getenv("GATEWAY_PORT", "8100")))
