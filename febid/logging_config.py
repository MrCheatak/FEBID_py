import logging
from tqdm import tqdm

class TqdmLoggingHandler(logging.Handler):
    """Logging handler that writes records through ``tqdm``-safe output."""

    def emit(self, record):
        """Emit a single log record using ``tqdm.write``.

        :param record: Log record to format and output.
        :type record: logging.LogRecord
        :return: None
        """
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)

def setup_logger(name="febid", level=logging.INFO):
    """Create or return a configured logger for the FEBID application.

    :param name: Logger name.
    :type name: str
    :param level: Logging level to assign when creating the logger.
    :type level: int
    :return: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = TqdmLoggingHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger

