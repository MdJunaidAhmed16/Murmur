"""Entry point: python -m murmur.web."""

import uvicorn

from murmur.config import get_settings
from murmur.logs import configure_logging
from murmur.web.app import create_app


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_config=None)


if __name__ == "__main__":
    main()
