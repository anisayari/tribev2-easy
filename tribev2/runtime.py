from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import warnings

_WARNING_FILTERS: tuple[tuple[str, type[Warning] | None], ...] = (
    (r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated.*", FutureWarning),
    (r"LabelEncoder: event_types has not been set.*", UserWarning),
    (
        r"The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
        FutureWarning,
    ),
    (r"The events dataframe contains an `Index` column.*", UserWarning),
    (r"LabelEncoder has only found one label.*", UserWarning),
    (
        r"Accessing `__path__` from `\.models\..*image_processing_.*`\..*alias will be removed in future versions\.",
        None,
    ),
)

_WARNING_LOGGER_INSTALLED = False
_TQDM_DESTRUCTOR_PATCHED = False


def apply_warning_filters() -> None:
    for message, category in _WARNING_FILTERS:
        kwargs = {"message": message}
        if category is not None:
            kwargs["category"] = category
        warnings.filterwarnings("ignore", **kwargs)
    logging.getLogger("neuralset.extractors.base").setLevel(logging.ERROR)
    _install_tqdm_destructor_guard()


def configure_file_logging(
    log_file: str | Path,
    *,
    logger_name: str = "tribev2",
    level: int = logging.INFO,
) -> Path:
    path = Path(log_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(
        isinstance(handler, RotatingFileHandler)
        and Path(getattr(handler, "baseFilename", "")).resolve() == path
        for handler in logger.handlers
    ):
        handler = RotatingFileHandler(
            path,
            maxBytes=2_500_000,
            backupCount=4,
            encoding="utf-8",
        )
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)

    _install_warning_logger()
    return path


def _install_warning_logger() -> None:
    global _WARNING_LOGGER_INSTALLED
    if _WARNING_LOGGER_INSTALLED:
        return

    original_showwarning = warnings.showwarning

    def _showwarning(
        message,
        category,
        filename,
        lineno,
        file=None,
        line=None,
    ):
        logging.getLogger("tribev2.warnings").warning(
            "%s:%s | %s | %s",
            filename,
            lineno,
            getattr(category, "__name__", str(category)),
            message,
        )
        if file is not None:
            original_showwarning(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = _showwarning
    _WARNING_LOGGER_INSTALLED = True


def _install_tqdm_destructor_guard() -> None:
    global _TQDM_DESTRUCTOR_PATCHED
    if _TQDM_DESTRUCTOR_PATCHED:
        return
    try:
        from tqdm.std import tqdm as tqdm_cls
    except Exception:
        return

    original_del = getattr(tqdm_cls, "__del__", None)
    if original_del is None:
        _TQDM_DESTRUCTOR_PATCHED = True
        return

    def _safe_del(self) -> None:
        if not hasattr(self, "last_print_t"):
            return
        try:
            original_del(self)
        except AttributeError as exc:
            if "last_print_t" not in str(exc):
                raise

    tqdm_cls.__del__ = _safe_del
    _TQDM_DESTRUCTOR_PATCHED = True
