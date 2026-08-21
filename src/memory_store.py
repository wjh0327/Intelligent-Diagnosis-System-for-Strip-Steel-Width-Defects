"""
带钢宽度缺陷诊断系统 —— 长期记忆存储模块
=========================================
三层降级策略（自动选择可用后端）：
  1. Redis   ：首选。连接成功即启用，所有记忆写入 Redis（带 TTL）。
  2. 本地文件：Redis 不可用时，落到项目 data/memory/ 下的 JSON 文件，
               同样支持跨进程/跨重启持久化。
  3. 内存    ：以上都不可用时，退化为进程内 dict（与旧版行为一致）。

存储内容：
  - 会话消息      rag:session:{sid}:messages        （最近 MAX_STORED_MESSAGES 条）
  - 批量诊断结果  rag:session:{sid}:batch
  - 单卷诊断      rag:diag:{feat_hash}              （按特征哈希，便于长期追溯）
  - 检索缓存      rag:cache:retrieve:{sha1}         （支持知识库更新后批量失效）

环境变量：
  RAG_REDIS_URL            默认 redis://127.0.0.1:6379/0
  RAG_MEMORY_TTL_SECONDS   默认 2592000（30 天）
  RAG_MAX_STORED_MESSAGES  默认 100
  RAG_MEMORY_DIR           文件降级目录（默认 <项目根>/data/memory）
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("RAG_REDIS_URL", "redis://127.0.0.1:6379/0")
MEMORY_TTL = int(os.getenv("RAG_MEMORY_TTL_SECONDS", str(30 * 24 * 3600)))
MAX_STORED_MESSAGES = int(os.getenv("RAG_MAX_STORED_MESSAGES", "100"))
ENV_MEMORY_DIR = os.getenv("RAG_MEMORY_DIR", "").strip()

_SID_RE = re.compile(r"^[0-9a-f]{32}$")
_CACHE_PREFIX = "rag:cache:retrieve:"


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class MemoryStore:
    """长期记忆统一入口：Redis -> 本地文件 -> 内存。"""

    def __init__(self, url: str = REDIS_URL, memory_dir: Optional[str] = None) -> None:
        self._url = url
        self._client = None
        self._local: Dict[str, Any] = {}
        self._dir: Optional[Path] = None

        # ---- 1. Redis 后端 ----
        try:
            import redis  # type: ignore
            from redis.backoff import ExponentialWithJitterBackoff
            from redis.retry import Retry

            client = redis.Redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=2,
                retry=Retry(ExponentialWithJitterBackoff(0.01, 0.01), 0),
            )
            client.ping()
            self._client = client
            logger.info("长期记忆：Redis 已启用 (%s)", url)
        except Exception as e:  # noqa: BLE001
            self._client = None
            logger.warning("长期记忆：Redis 不可用，尝试文件降级 (%s)", e)

        # ---- 2. 文件后端 ----
        candidates: List[str] = []
        if memory_dir:
            candidates.append(memory_dir)
        elif ENV_MEMORY_DIR:
            candidates.append(ENV_MEMORY_DIR)
        elif self._client is None:
            # 默认放项目 data/memory（Redis 可用时不创建，避免无谓写盘）
            candidates.append(str(Path(__file__).resolve().parent.parent / "data" / "memory"))
        for cand in candidates:
            try:
                p = Path(cand)
                p.mkdir(parents=True, exist_ok=True)
                probe = p / ".write_probe"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink(missing_ok=True)
                self._dir = p
                break
            except Exception as e:  # noqa: BLE001
                logger.warning("长期记忆：文件目录不可写 %s (%s)", cand, e)

        if self._client is None and self._dir is None:
            logger.warning("长期记忆：Redis 与文件后端均不可用，退化为进程内内存")

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def mode(self) -> str:
        if self._client is not None:
            return "redis"
        if self._dir is not None:
            return "file"
        return "memory"

    @property
    def enabled(self) -> bool:
        return self.mode != "memory"

    def status(self) -> Dict[str, Any]:
        return {"mode": self.mode, "url": self._url, "dir": str(self._dir) if self._dir else None}

    @staticmethod
    def is_valid_sid(sid: str) -> bool:
        return bool(sid and _SID_RE.match(sid))

    def _expired(self, ts: float) -> bool:
        return (time.time() - ts) > MEMORY_TTL

    # ------------------------------------------------------------------
    # 会话消息
    # ------------------------------------------------------------------
    def get_messages(self, sid: str) -> List[Dict[str, str]]:
        if not self.is_valid_sid(sid):
            return []
        if self._client is not None:
            try:
                raw = self._client.get(f"rag:session:{sid}:messages")
                if raw:
                    data = json.loads(raw)
                    if isinstance(data, list):
                        return [m for m in data if isinstance(m, dict)]
            except Exception as e:  # noqa: BLE001
                logger.warning("读取会话消息失败: %s", e)
            return []
        if self._dir is not None:
            data = self._read_json(self._dir / f"{sid}.json")
            if data and not self._expired(data.get("updated_at", 0)):
                msgs = data.get("messages", [])
                return [m for m in msgs if isinstance(m, dict)]
            return []
        return list(self._local.get(f"m:{sid}", []))

    def save_messages(self, sid: str, messages: List[Dict[str, str]]) -> None:
        if not self.is_valid_sid(sid):
            return
        capped = messages[-MAX_STORED_MESSAGES:]
        if self._client is not None:
            try:
                self._client.set(
                    f"rag:session:{sid}:messages",
                    json.dumps(capped, ensure_ascii=False),
                    ex=MEMORY_TTL,
                )
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("写入会话消息失败: %s", e)
        if self._dir is not None:
            data = self._read_json(self._dir / f"{sid}.json") or {}
            data["messages"] = capped
            data["updated_at"] = time.time()
            self._write_json(self._dir / f"{sid}.json", data)
            return
        self._local[f"m:{sid}"] = capped

    # ------------------------------------------------------------------
    # 批量诊断结果
    # ------------------------------------------------------------------
    def get_batch(self, sid: str) -> Dict[str, Any]:
        if not self.is_valid_sid(sid):
            return {}
        if self._client is not None:
            try:
                raw = self._client.get(f"rag:session:{sid}:batch")
                if raw:
                    data = json.loads(raw)
                    return data if isinstance(data, dict) else {}
            except Exception as e:  # noqa: BLE001
                logger.warning("读取批量结果失败: %s", e)
            return {}
        if self._dir is not None:
            data = self._read_json(self._dir / f"{sid}.json")
            if data and not self._expired(data.get("updated_at", 0)):
                batch = data.get("batch")
                return batch if isinstance(batch, dict) else {}
            return {}
        return dict(self._local.get(f"b:{sid}", {}))

    def save_batch(self, sid: str, batch: Dict[str, Any]) -> None:
        if not self.is_valid_sid(sid):
            return
        if self._client is not None:
            try:
                self._client.set(
                    f"rag:session:{sid}:batch",
                    json.dumps(batch, ensure_ascii=False),
                    ex=MEMORY_TTL,
                )
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("写入批量结果失败: %s", e)
        if self._dir is not None:
            data = self._read_json(self._dir / f"{sid}.json") or {}
            data["batch"] = batch
            data["updated_at"] = time.time()
            self._write_json(self._dir / f"{sid}.json", data)
            return
        self._local[f"b:{sid}"] = dict(batch)

    def clear_session(self, sid: str) -> None:
        if not self.is_valid_sid(sid):
            return
        if self._client is not None:
            try:
                pipe = self._client.pipeline()
                pipe.delete(f"rag:session:{sid}:messages")
                pipe.delete(f"rag:session:{sid}:batch")
                pipe.execute()
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("清除会话失败: %s", e)
        if self._dir is not None:
            try:
                (self._dir / f"{sid}.json").unlink(missing_ok=True)
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("清除会话文件失败: %s", e)
        self._local.pop(f"m:{sid}", None)
        self._local.pop(f"b:{sid}", None)

    # ------------------------------------------------------------------
    # 单卷诊断（按 12 维特征哈希索引，长期追溯）
    # ------------------------------------------------------------------
    def save_diag(self, feat_hash: str, data: Dict[str, Any]) -> None:
        if not feat_hash:
            return
        record = dict(data)
        record["updated_at"] = time.time()
        if self._client is not None:
            try:
                self._client.set(
                    f"rag:diag:{feat_hash}",
                    json.dumps(record, ensure_ascii=False),
                    ex=MEMORY_TTL,
                )
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("写入单卷诊断失败: %s", e)
        if self._dir is not None:
            self._write_json(self._dir / "diag" / f"{feat_hash}.json", record)
            return
        self._local[f"d:{feat_hash}"] = record

    def get_diag(self, feat_hash: str) -> Optional[Dict[str, Any]]:
        if not feat_hash:
            return None
        if self._client is not None:
            try:
                raw = self._client.get(f"rag:diag:{feat_hash}")
                if raw:
                    data = json.loads(raw)
                    return data if isinstance(data, dict) else None
            except Exception as e:  # noqa: BLE001
                logger.warning("读取单卷诊断失败: %s", e)
            return None
        if self._dir is not None:
            return self._read_json(self._dir / "diag" / f"{feat_hash}.json")
        return dict(self._local.get(f"d:{feat_hash}", {})) or None

    def get_diag_by_features(self, features: List[float]) -> Optional[Dict[str, Any]]:
        """按 12 维特征反查历史诊断记录（兼容 int/float 表示差异导致的哈希不一致）。"""
        if not features or len(features) != 12:
            return None
        candidates = [list(features)]
        try:
            candidates.append(
                [int(f) if float(f).is_integer() else float(f) for f in features]
            )
        except Exception:  # noqa: BLE001
            pass
        candidates.append([float(f) for f in features])
        for cand in candidates:
            feat_hash = hashlib.md5(
                json.dumps(cand, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            rec = self.get_diag(feat_hash)
            if rec:
                return rec
        return None

    # ------------------------------------------------------------------
    # 检索缓存
    # ------------------------------------------------------------------
    def cache_get(self, key: str) -> Optional[Any]:
        if self._client is not None:
            try:
                raw = self._client.get(key)
                if raw:
                    return json.loads(raw)
            except Exception as e:  # noqa: BLE001
                logger.warning("读取检索缓存失败: %s", e)
            return None
        if self._dir is not None:
            record = self._read_json(self._dir / "cache" / f"retrieve_{_sha1(key)}.json")
            if record and record.get("expires_at", 0) > time.time():
                return record.get("value")
            return None
        return self._local.get(key)

    def cache_set(self, key: str, value: Any, ttl: int = 86400) -> None:
        if self._client is not None:
            try:
                self._client.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("写入检索缓存失败: %s", e)
        if self._dir is not None:
            record = {"expires_at": time.time() + ttl, "value": value}
            self._write_json(self._dir / "cache" / f"retrieve_{_sha1(key)}.json", record)
            return
        self._local[key] = value

    def invalidate_retrieve_cache(self) -> int:
        """清除全部检索缓存（知识库更新后调用），返回清除条数。"""
        n = 0
        if self._client is not None:
            try:
                keys = list(self._client.scan_iter(f"{_CACHE_PREFIX}*"))
                if keys:
                    n = self._client.delete(*keys)
                return n
            except Exception as e:  # noqa: BLE001
                logger.warning("失效检索缓存失败: %s", e)
                return 0
        if self._dir is not None:
            cache_dir = self._dir / "cache"
            if cache_dir.is_dir():
                try:
                    for f in cache_dir.glob("retrieve_*.json"):
                        f.unlink(missing_ok=True)
                        n += 1
                except Exception as e:  # noqa: BLE001
                    logger.warning("失效检索缓存文件失败: %s", e)
            return n
        stale = [k for k in self._local if k.startswith(_CACHE_PREFIX)]
        for k in stale:
            self._local.pop(k, None)
            n += 1
        return n

    # ------------------------------------------------------------------
    # 文件读写工具
    # ------------------------------------------------------------------
    def _read_json(self, path: Path) -> Optional[Any]:
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning("读取文件失败 %s: %s", path, e)
        return None

    def _write_json(self, path: Path, data: Any) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        except Exception as e:  # noqa: BLE001
            logger.warning("写入文件失败 %s: %s", path, e)


_store: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    """进程内单例。"""
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store
