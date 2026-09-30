# -*- coding: utf-8 -*-
"""
src/logger.py —— 统一日志配置
==============================
输出到控制台与 logs/app.log，按模块命名 logger。
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_configured = False

# 单文件 10MB、保留 5 份备份：磁盘占用上限约 60MB，长驻服务日志不再无界增长
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5


class _SafeRotatingFileHandler(RotatingFileHandler):
    """Windows 下 Streamlit 与 uvicorn 可能同时持有 app.log：一方 rollover 的
    rename 会因另一方占用而失败，超类此时已 close 且未重开，后续日志会
    永久丢失。这里吞掉本轮轮转、以追加模式重开原文件继续写——轮转失败的
    代价退化为单文件超限增长，而不是日志停写。"""

    def doRollover(self):
        try:
            super().doRollover()
        except OSError:
            if self.stream is None or self.stream.closed:
                self.stream = self._open()


def setup_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    _configured = True
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)
    try:
        file_handler = _SafeRotatingFileHandler(
            LOG_DIR / "app.log",
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except Exception:
        pass


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
