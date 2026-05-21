"""Small logging helpers for user-facing package output."""

import logging


def log_or_print(logger: logging.Logger, message: str, *args: object) -> None:
    """Log an info message, falling back to print when logging is not configured."""
    if logger.isEnabledFor(logging.INFO):
        logger.info(message, *args)
    else:
        print(message % args if args else message)
