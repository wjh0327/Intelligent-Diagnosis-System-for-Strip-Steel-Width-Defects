"""
带钢宽度缺陷诊断系统 —— 长期记忆存储模块
=========================================
存储降级策略（自动选择可用后端）：
  1. Redis   ：会话消息 / 批量诊断 / 检索缓存的首选后端（带 TTL）。
  2. MySQL   ：单卷诊断记录的结构化后端（可选）。启用后可按故障类型、
                时间范围做 SQL 查询；不可用时自动回落到 Redis/文件。
  3. 本地文件：以上不可用时，落到项目 data/memory/ 下的 JSON 文件。
  4. 内存    ：全部不可用时退化为进程内 dict（与旧版行为一致）。

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
  RAG_MYSQL_HOST           默认空（不启用 MySQL）；启用示例 127.0.0.1
  RAG_MYSQL_PORT           默认 3306
  RAG_MYSQL_USER           默认 rag
  RAG_MYSQL_PASSWORD       默认 rag123456
  RAG_MYSQL_DB             默认 rag_agent（表 rag_diag 自动创建）
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("RAG_REDIS_URL", "redis://127.0.0.1:6379/0")
MEMORY_TTL = int(os.getenv("RAG_MEMORY_TTL_SECONDS", str(30 * 24 * 3600)))
MAX_STORED_MESSAGES = int(os.getenv("RAG_MAX_STORED_MESSAGES", "100"))
ENV_MEMORY_DIR = os.getenv("RAG_MEMORY_DIR", "").strip()
MYSQL_HOST = os.getenv("RAG_MYSQL_HOST", "").strip()
MYSQL_PORT = int(os.getenv("RAG_MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("RAG_MYSQL_USER", "rag")
MYSQL_PASSWORD = os.getenv("RAG_MYSQL_PASSWORD", "rag123456")
MYSQL_DB = os.getenv("RAG_MYSQL_DB", "rag_agent")

_SID_RE = re.compile(r"^[0-9a-f]{32}$")
_CACHE_PREFIX = "rag:cache:retrieve:"

_MYSQL_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rag_diag (
  feat_hash   CHAR(32)     NOT NULL PRIMARY KEY,
  fault_cn    VARCHAR(64)  NOT NULL,
  fault_desc  TEXT         NULL,
  confidence  DOUBLE       NULL,
  features    JSON         NULL,
  updated_at  DOUBLE       NOT NULL,
  KEY idx_fault_cn (fault_cn),
  KEY idx_updated_at (updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class MemoryStore:
    """长期记忆统一入口：Redis -> 本地文件 -> 内存。"""

    def __init__(self, url: str = REDIS_URL, memory_dir: Optional[str] = None) -> None:
        self._url = url
        self._client = None
        self._local: Dict[str, Any] = {}
        self._dir: Optional[Path] = None
        self._mysql = None
        self._mysql_lock = threading.Lock()
        # 检索缓存失效回调（由调用方注册，用于同步清理其进程内缓存）
        self._invalidate_callbacks: List[Any] = []

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

        # ---- 3. MySQL 后端（单卷诊断记录结构化存储，可选） ----
        if MYSQL_HOST:
            try:
                import pymysql  # type: ignore

                conn = pymysql.connect(
                    host=MYSQL_HOST,
                    port=MYSQL_PORT,
                    user=MYSQL_USER,
                    password=MYSQL_PASSWORD,
                    database=MYSQL_DB,
                    charset="utf8mb4",
                    connect_timeout=3,
                    autocommit=True,
                )
                with self._mysql_lock:
                    with conn.cursor() as cur:
                        cur.execute(_MYSQL_SCHEMA_SQL)
                self._mysql = conn
                logger.info(
                    "长期记忆：MySQL 已启用（诊断记录结构化存储） %s:%s/%s",
                    MYSQL_HOST, MYSQL_PORT, MYSQL_DB,
                )
            except Exception as e:  # noqa: BLE001
                self._mysql = None
                logger.warning(
                    "长期记忆：MySQL 不可用，诊断记录继续使用 Redis/文件降级 (%s)",
                    e,
                )

        if self._client is None and self._dir is None:
            logger.warning("长期记忆：Redis 与文件后端均不可用，退化为进程内内存")

        if self._dir is not None:
            self._sweep_expired_files()

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
        return {
            "mode": self.mode,
            "url": self._url,
            "dir": str(self._dir) if self._dir else None,
            "mysql": self._mysql is not None,
        }

    @staticmethod
    def is_valid_sid(sid: str) -> bool:
        return bool(sid and _SID_RE.match(sid))

    def register_invalidate_callback(self, fn) -> None:
        """注册"检索缓存已失效"回调，供调用方同步清理其进程内缓存。

        回调应返回本次清空的条目数（可返回 0/None）。注册是幂等的。
        示例：service 模块用它清理 _retrieve_cache 与相似案例缓存。
        """
        if fn not in self._invalidate_callbacks:
            self._invalidate_callbacks.append(fn)

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
            # 未命中/失败不在此短路：Redis 抖动期的写入会落到进程内 _local
            # （Redis 就绪时文件后端未启用），读必须继续查下级，
            # 否则抖动期数据在本进程内不可见——Streamlit 刷新即"对话消失"
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
            # 同 get_messages：未命中/失败继续查下级，避免抖动期批量状态回退
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
        if self._mysql is not None:
            try:
                with self._mysql_lock:
                    self._mysql.ping(reconnect=True)
                    with self._mysql.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO rag_diag
                              (feat_hash, fault_cn, fault_desc, confidence, features, updated_at)
                            VALUES (%s,%s,%s,%s,%s,%s)
                            ON DUPLICATE KEY UPDATE
                              fault_cn=VALUES(fault_cn),
                              fault_desc=VALUES(fault_desc),
                              confidence=VALUES(confidence),
                              features=VALUES(features),
                              updated_at=VALUES(updated_at)
                            """,
                            (
                                feat_hash,
                                record.get("fault_cn"),
                                record.get("fault_desc"),
                                record.get("confidence"),
                                json.dumps(record.get("features"), ensure_ascii=False),
                                record.get("updated_at"),
                            ),
                        )
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("写入 MySQL 单卷诊断失败，回退 Redis/文件: %s", e)
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
            # 文件模式与 Redis 对齐 TTL：包一层 expires_at，读取时校验，
            # 过期文件由启动扫描 _sweep_expired_files 与读取路径双向清理
            self._write_json(
                self._dir / "diag" / f"{feat_hash}.json",
                {"expires_at": time.time() + MEMORY_TTL, "record": record},
            )
            return
        # 进程内兜底同样带过期（纯内存模式防长驻进程无界累积）
        self._local[f"d:{feat_hash}"] = (time.time() + MEMORY_TTL, record)

    def get_diag(self, feat_hash: str) -> Optional[Dict[str, Any]]:
        if not feat_hash:
            return None
        if self._mysql is not None:
            try:
                with self._mysql_lock:
                    self._mysql.ping(reconnect=True)
                    with self._mysql.cursor() as cur:
                        cur.execute(
                            """
                            SELECT feat_hash, fault_cn, fault_desc, confidence, features, updated_at
                            FROM rag_diag WHERE feat_hash=%s
                            """,
                            (feat_hash,),
                        )
                        row = cur.fetchone()
                if row:
                    return {
                        "feat_hash": row[0],
                        "fault_cn": row[1],
                        "fault_desc": row[2],
                        "confidence": row[3],
                        "features": json.loads(row[4]) if row[4] else None,
                        "updated_at": row[5],
                    }
            except Exception as e:  # noqa: BLE001
                logger.warning("读取 MySQL 单卷诊断失败，回退 Redis/文件: %s", e)
        if self._client is not None:
            try:
                raw = self._client.get(f"rag:diag:{feat_hash}")
                if raw:
                    data = json.loads(raw)
                    return data if isinstance(data, dict) else None
            except Exception as e:  # noqa: BLE001
                logger.warning("读取单卷诊断失败: %s", e)
            # 同 get_messages：未命中/失败继续查下级
        if self._dir is not None:
            path = self._dir / "diag" / f"{feat_hash}.json"
            wrapped = self._read_json(path)
            if not wrapped:
                return None
            if isinstance(wrapped, dict) and "record" in wrapped:
                # 新格式：带 expires_at 的包装记录
                if wrapped.get("expires_at", 0) <= time.time():
                    path.unlink(missing_ok=True)
                    return None
                return wrapped["record"]
            # 旧格式裸记录：按 updated_at 判过期
            if (wrapped.get("updated_at") or 0) + MEMORY_TTL <= time.time():
                path.unlink(missing_ok=True)
                return None
            return wrapped
        entry = self._local.get(f"d:{feat_hash}")
        if isinstance(entry, tuple) and len(entry) == 2:
            expires_at, record = entry
            if expires_at > time.time():
                return dict(record)
            self._local.pop(f"d:{feat_hash}", None)
            return None
        return None

    def get_diag_history(
        self,
        fault_cn: Optional[str] = None,
        days: Optional[int] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """按故障类型/时间范围查询单卷诊断记录（需 MySQL 后端）。"""
        if self._mysql is None:
            return []
        sql = (
            "SELECT feat_hash, fault_cn, fault_desc, confidence, features, updated_at "
            "FROM rag_diag WHERE 1=1"
        )
        args: List[Any] = []
        if fault_cn:
            sql += " AND fault_cn=%s"
            args.append(fault_cn)
        if days:
            sql += " AND updated_at >= %s"
            args.append(time.time() - days * 86400)
        sql += " ORDER BY updated_at DESC LIMIT %s"
        args.append(int(limit))
        try:
            with self._mysql_lock:
                self._mysql.ping(reconnect=True)
                with self._mysql.cursor() as cur:
                    cur.execute(sql, tuple(args))
                    rows = cur.fetchall()
        except Exception as e:  # noqa: BLE001
            logger.warning("查询 MySQL 诊断记录失败: %s", e)
            return []
        if not rows:
            return []
        return [
            {
                "feat_hash": r[0],
                "fault_cn": r[1],
                "fault_desc": r[2],
                "confidence": r[3],
                "features": json.loads(r[4]) if r[4] else None,
                "updated_at": r[5],
            }
            for r in rows
        ]

    def _sweep_expired_files(self) -> None:
        """文件模式启动时清理过期的单卷诊断记录。

        Redis 的过期由服务端 TTL 完成，文件模式必须自行清扫，否则未被
        再次读取的过期文件会永远留在盘上。按 mtime 判定（mtime 即写入
        时刻，与 expires_at 一致），避免逐个解析 JSON。"""
        diag_dir = self._dir / "diag"
        if not diag_dir.is_dir():
            return
        now = time.time()
        removed = 0
        for f in diag_dir.glob("*.json"):
            try:
                if now - f.stat().st_mtime > MEMORY_TTL:
                    f.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                continue
        if removed:
            logger.info("已清理 %d 条过期的文件模式单卷诊断记录", removed)

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
            # 未命中/失败不在此短路：Redis 只读/maxmemory 期间 cache_set 的写入
            # 落在 _local 兜底，读必须穿透否则缓存完全失效（每次请求全量检索）。
            # 穿透的安全性由 invalidate_retrieve_cache 跨后端清理保证
        if self._dir is not None:
            record = self._read_json(self._dir / "cache" / f"retrieve_{_sha1(key)}.json")
            if record and record.get("expires_at", 0) > time.time():
                return record.get("value")
        entry = self._local.get(key)
        if isinstance(entry, tuple) and len(entry) == 2:
            expires_at, value = entry
            if expires_at > time.time():
                return value
            self._local.pop(key, None)  # 过期立即逐出
        return None

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
        # 进程内兜底（Redis 写失败期间的实际存储 / 纯内存模式的主存储），
        # 带过期时间供 cache_get 校验
        self._local[key] = (time.time() + ttl, value)

    def invalidate_retrieve_cache(self) -> int:
        """清除全部检索缓存（知识库更新后调用），返回清除条数。

        除当前后端的缓存外，还会触发 register_invalidate_callback 注册的回调，
        让调用方进程内的检索缓存（如 service 的 _retrieve_cache / 相似案例缓存）
        同步失效——否则更新知识库后仍会命中进程内的旧结果。
        """
        n = 0
        if self._client is not None:
            try:
                keys = list(self._client.scan_iter(f"{_CACHE_PREFIX}*"))
                if keys:
                    n += self._client.delete(*keys)
            except Exception as e:  # noqa: BLE001
                logger.warning("失效检索缓存失败: %s", e)
        if self._dir is not None:
            cache_dir = self._dir / "cache"
            if cache_dir.is_dir():
                try:
                    for f in cache_dir.glob("retrieve_*.json"):
                        f.unlink(missing_ok=True)
                        n += 1
                except Exception as e:  # noqa: BLE001
                    logger.warning("失效检索缓存文件失败: %s", e)
        # _local 恒清理（cache_get 读穿透依赖此保证）：抖动期写入 _local 的
        # 条目不因 Redis 键删除而消失，失效必须覆盖全部可读后端
        for k in [k for k in self._local if k.startswith(_CACHE_PREFIX)]:
            self._local.pop(k, None)
            n += 1
        # 无论命中哪个后端，都要通知进程内缓存失效（Redis 就绪时进程内缓存通常为空，
        # 调用是幂等的）
        for fn in list(self._invalidate_callbacks):
            try:
                n += int(fn() or 0)
            except Exception as e:  # noqa: BLE001
                logger.warning("检索缓存失效回调执行失败: %s", e)
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
