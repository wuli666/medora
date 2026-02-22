import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from src.config.settings import settings

_BASE_LOGGER_NAME = "uvicorn.error"
_CONFIGURED = False
_DEBUG_FILE_HANDLER_MARK = "_medgemma_debug_file"


def _resolve_log_level(level_name: str) -> int:
    level = getattr(logging, (level_name or "").strip().upper(), None)
    if isinstance(level, int):
        return level
    return logging.INFO


def _is_valid_log_level(level_name: str) -> bool:
    return isinstance(getattr(logging, (level_name or "").strip().upper(), None), int)


def _ensure_debug_file_handler(base_logger: logging.Logger) -> None:
    for handler in base_logger.handlers:
        if getattr(handler, _DEBUG_FILE_HANDLER_MARK, False):
            return

    backup_count = settings.LOG_FILE_BACKUP_COUNT
    if backup_count < 0:
        base_logger.warning(
            "[logger] Invalid LOG_FILE_BACKUP_COUNT '%s', fallback to 7",
            backup_count,
        )
        backup_count = 7

    file_level = (
        _resolve_log_level(settings.LOG_FILE_LEVEL)
        if _is_valid_log_level(settings.LOG_FILE_LEVEL)
        else logging.DEBUG
    )
    if not _is_valid_log_level(settings.LOG_FILE_LEVEL):
        base_logger.warning(
            "[logger] Invalid LOG_FILE_LEVEL '%s', fallback to DEBUG",
            settings.LOG_FILE_LEVEL,
        )

    try:
        log_dir = Path(settings.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / settings.LOG_FILE_NAME

        file_handler = TimedRotatingFileHandler(
            filename=str(log_path),
            when=settings.LOG_FILE_WHEN,
            interval=settings.LOG_FILE_INTERVAL,
            backupCount=backup_count,
            encoding=settings.LOG_FILE_ENCODING,
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))
        setattr(file_handler, _DEBUG_FILE_HANDLER_MARK, True)
        base_logger.addHandler(file_handler)
    except Exception as exc:
        base_logger.warning(
            "[logger] Failed to configure debug file logging at '%s': %s",
            settings.LOG_DIR,
            exc,
        )


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    base_logger = logging.getLogger(_BASE_LOGGER_NAME)
    log_level = (
        _resolve_log_level(settings.LOG_LEVEL)
        if _is_valid_log_level(settings.LOG_LEVEL)
        else logging.INFO
    )

    if base_logger.handlers:
        base_logger.setLevel(log_level)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))
        base_logger.addHandler(handler)
        base_logger.setLevel(log_level)
        base_logger.propagate = False

    if not _is_valid_log_level(settings.LOG_LEVEL):
        base_logger.warning(
            "[logger] Invalid LOG_LEVEL '%s', fallback to INFO",
            settings.LOG_LEVEL,
        )

    _ensure_debug_file_handler(base_logger)

    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    configure_logging()
    base_logger = logging.getLogger(_BASE_LOGGER_NAME)
    if not name:
        return base_logger
    return base_logger.getChild(name)


def log_stage(logger: logging.Logger, stage: str, content: str) -> None:
    truncated = ""

    if not content:
        truncated = "[EMPTY]"

    if len(content) <= settings.AGENT_LOG_TRUNCATE:
        truncated = content
    else:
        truncated = (
            f"{content[:settings.AGENT_LOG_TRUNCATE]} "
            f"...[truncated {len(content) - settings.AGENT_LOG_TRUNCATE} chars]"
        )

    logger.info("[%s] output:\n%s", stage, truncated)
