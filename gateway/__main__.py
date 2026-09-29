import os
import sys
from pathlib import Path

import uvicorn

from gateway.app import app_from_env
from gateway.store import Store


def main(argv: list[str]) -> None:
    """Serve the gateway, or ``backup <file>`` to copy its database safely while it runs."""
    if argv[:1] == ["backup"] and len(argv) == 2:
        Store(Path(os.getenv("GATEWAY_DB", "gateway-data/gateway.sqlite"))).backup(argv[1])
        return
    if argv:
        sys.exit("usage: python -m gateway [backup <file>]")
    uvicorn.run(
        app_from_env(),
        host=os.getenv("GATEWAY_HOST", "127.0.0.1"),
        port=int(os.getenv("GATEWAY_PORT", "8100")),
        # Only the reverse proxy named here may set the client address that the per-network limits count.
        proxy_headers=True,
        forwarded_allow_ips=os.getenv("GATEWAY_TRUSTED_PROXIES", "127.0.0.1"),
        server_header=False,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
