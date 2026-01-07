import logging
from pathlib import Path

def setup_logger(name: str = "rag", log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger
