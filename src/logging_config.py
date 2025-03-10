import logging
import logging.config
import os

# Ensure logs directory exists
LOG_DIR = "../logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "simple": {
            "format": "%(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "INFO"
        },
        "file_rag_data_management": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": os.path.join(LOG_DIR, "rag_data_management.log"),
            "maxBytes": 5 * 1024 * 1024,  # 5 MB
            "backupCount": 5,
            "formatter": "detailed",
            "level": "INFO"
        },
        "file_query_logs": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": os.path.join(LOG_DIR, "query_logs.log"),
            "maxBytes": 5 * 1024 * 1024,  # 5 MB
            "backupCount": 5,
            "formatter": "detailed",
            "level": "INFO"
        }
    },
    "loggers": {
        "root": {
            "level": "DEBUG",
            "handlers": ["console"]
        },
        "rag_data_management_logger": {
            "level": "INFO",
            "handlers": ["console", "file_rag_data_management"],
            "propagate": False
        }
    }
}


def setup_logging():
    """Set up logging configuration."""
    logging.config.dictConfig(LOGGING_CONFIG)
