"""Claude Code 风格的本地会话 Transcript 与上下文快照存储。"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)
from loguru import logger

TRANSCRIPT_VERSION = 1
SUMMARY_SOURCE = "summarization"


def default_conversation_data_dir(configured_path: str = "") -> Path:
    """返回当前操作系统用户的默认会话目录。"""
    if configured_path.strip():
        return Path(configured_path).expanduser().resolve()

    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "OnCall" / "sessions"

    xdg_data_home = os.getenv("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / "oncall" / "sessions"

    return Path.home() / ".local" / "share" / "oncall" / "sessions"


def _utc_timestamp() -> str:
    return datetime.now().astimezone().isoformat()


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        return "".join(text_parts)
    return str(content)


class ConversationTranscriptStore:
    """按会话追加保存完整消息，并保存版本化 Compaction 快照。"""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self._last_sequence_cache: dict[str, int] = {}

    @staticmethod
    def _session_key(session_id: str) -> str:
        """不直接使用外部 session_id 作为路径，避免路径穿越。"""
        return hashlib.sha256(session_id.encode("utf-8")).hexdigest()

    def session_dir(self, session_id: str, *, create: bool = False) -> Path:
        path = self.root_dir / self._session_key(session_id)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def transcript_path(self, session_id: str, *, create: bool = False) -> Path:
        return self.session_dir(session_id, create=create) / "transcript.jsonl"

    def _meta_path(self, session_id: str, *, create: bool = False) -> Path:
        return self.session_dir(session_id, create=create) / "meta.json"

    def _snapshots_dir(self, session_id: str, *, create: bool = False) -> Path:
        path = self.session_dir(session_id, create=create) / "snapshots"
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)

    def _read_records(self, session_id: str) -> list[dict[str, Any]]:
        path = self.transcript_path(session_id)
        if not path.exists():
            return []

        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    logger.warning(
                        "会话 {} 的 Transcript 第 {} 行不完整，已跳过",
                        session_id,
                        line_number,
                    )
                    continue
                if not isinstance(record, dict):
                    continue
                if record.get("session_id") != session_id:
                    logger.warning(
                        "会话 {} 的 Transcript 第 {} 行标识不匹配，已跳过",
                        session_id,
                        line_number,
                    )
                    continue
                records.append(record)
        return records

    def last_sequence(self, session_id: str) -> int:
        cached = self._last_sequence_cache.get(session_id)
        if cached is not None:
            return cached
        records = self._read_records(session_id)
        sequence = max((int(record.get("seq", 0)) for record in records), default=0)
        self._last_sequence_cache[session_id] = sequence
        return sequence

    @staticmethod
    def _last_clear_sequence(records: Sequence[dict[str, Any]]) -> int:
        return max(
            (
                int(record.get("seq", 0))
                for record in records
                if record.get("record_type") == "clear"
            ),
            default=0,
        )

    def _update_meta(self, session_id: str, **updates: Any) -> None:
        path = self._meta_path(session_id, create=True)
        payload: dict[str, Any] = {
            "version": TRANSCRIPT_VERSION,
            "session_id": session_id,
            "created_at": _utc_timestamp(),
        }
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload.update(loaded)
            except (OSError, json.JSONDecodeError):
                logger.warning("会话 {} 的 meta.json 无法读取，将重建", session_id)
        payload.update(updates)
        payload["updated_at"] = _utc_timestamp()
        self._write_json_atomically(path, payload)

    def append_messages(
        self,
        session_id: str,
        messages: Iterable[BaseMessage],
    ) -> int:
        """将新消息追加到 Transcript，返回最后一个序号。"""
        message_list = list(messages)
        if not message_list:
            return self.last_sequence(session_id)

        for message in message_list:
            if message.id is None:
                message.id = str(uuid.uuid4())

        serialized_messages = messages_to_dict(message_list)
        next_sequence = self.last_sequence(session_id) + 1
        path = self.transcript_path(session_id, create=True)
        with path.open("a", encoding="utf-8", newline="\n") as file:
            for serialized in serialized_messages:
                record = {
                    "version": TRANSCRIPT_VERSION,
                    "seq": next_sequence,
                    "record_type": "message",
                    "session_id": session_id,
                    "timestamp": _utc_timestamp(),
                    "message": serialized,
                }
                file.write(
                    json.dumps(record, ensure_ascii=False, default=str) + "\n"
                )
                next_sequence += 1
            file.flush()
            os.fsync(file.fileno())

        last_sequence = next_sequence - 1
        self._last_sequence_cache[session_id] = last_sequence
        self._update_meta(session_id, last_sequence=last_sequence)
        return last_sequence

    def append_clear(self, session_id: str) -> int:
        """追加逻辑清空事件，保留原始 Transcript 以便审计和恢复。"""
        sequence = self.last_sequence(session_id) + 1
        path = self.transcript_path(session_id, create=True)
        record = {
            "version": TRANSCRIPT_VERSION,
            "seq": sequence,
            "record_type": "clear",
            "session_id": session_id,
            "timestamp": _utc_timestamp(),
        }
        with path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        self._last_sequence_cache[session_id] = sequence
        self._update_meta(session_id, last_sequence=sequence, cleared_at=_utc_timestamp())
        return sequence

    def load_messages(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
        include_summaries: bool = True,
    ) -> tuple[list[BaseMessage], int]:
        """读取指定序号之后的有效消息及当前 Transcript 最后序号。"""
        records = self._read_records(session_id)
        last_sequence = max(
            (int(record.get("seq", 0)) for record in records),
            default=0,
        )
        effective_after = max(after_sequence, self._last_clear_sequence(records))
        serialized: list[dict[str, Any]] = []
        for record in records:
            if record.get("record_type") != "message":
                continue
            if int(record.get("seq", 0)) <= effective_after:
                continue
            message_data = record.get("message")
            if not isinstance(message_data, dict):
                continue
            if not include_summaries:
                kwargs = message_data.get("data", {}).get("additional_kwargs", {})
                if kwargs.get("lc_source") == SUMMARY_SOURCE:
                    continue
            serialized.append(message_data)

        if not serialized:
            return [], last_sequence
        try:
            return messages_from_dict(serialized), last_sequence
        except Exception as exc:
            logger.warning("会话 {} 的消息反序列化失败: {}", session_id, exc)
            recovered: list[BaseMessage] = []
            for item in serialized:
                try:
                    recovered.extend(messages_from_dict([item]))
                except Exception:
                    continue
            return recovered, last_sequence

    def write_compaction_snapshot_if_changed(
        self,
        session_id: str,
        messages: Sequence[BaseMessage],
        transcript_sequence: int,
    ) -> bool:
        """仅在出现新的 Compaction 摘要时写入版本化快照。"""
        summary_message = next(
            (
                message
                for message in messages
                if (getattr(message, "additional_kwargs", {}) or {}).get(
                    "lc_source"
                )
                == SUMMARY_SOURCE
            ),
            None,
        )
        if summary_message is None:
            return False

        summary_message_id = str(summary_message.id or "")
        latest = self.load_latest_snapshot(session_id)
        if latest and latest.get("summary_message_id") == summary_message_id:
            return False

        snapshot_id = f"{transcript_sequence:020d}-{time.time_ns()}"
        payload = {
            "version": TRANSCRIPT_VERSION,
            "snapshot_id": snapshot_id,
            "session_id": session_id,
            "created_at": _utc_timestamp(),
            "transcript_sequence": transcript_sequence,
            "summary_message_id": summary_message_id,
            "messages": messages_to_dict(messages),
        }
        path = self._snapshots_dir(session_id, create=True) / f"{snapshot_id}.json"
        self._write_json_atomically(path, payload)
        self._update_meta(
            session_id,
            latest_snapshot=path.name,
            latest_snapshot_sequence=transcript_sequence,
        )
        return True

    def load_latest_snapshot(self, session_id: str) -> dict[str, Any] | None:
        snapshots_dir = self._snapshots_dir(session_id)
        if not snapshots_dir.exists():
            return None

        records = self._read_records(session_id)
        last_clear_sequence = self._last_clear_sequence(records)
        for path in sorted(snapshots_dir.glob("*.json"), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("会话 {} 的快照 {} 已损坏，尝试更早快照", session_id, path.name)
                continue
            if not isinstance(payload, dict) or payload.get("session_id") != session_id:
                continue
            if int(payload.get("transcript_sequence", 0)) < last_clear_sequence:
                continue
            if not isinstance(payload.get("messages"), list):
                continue
            return payload
        return None

    def load_recovery_messages(self, session_id: str) -> list[BaseMessage]:
        """加载最新快照和其后的增量消息，用于重建 MemorySaver。"""
        snapshot = self.load_latest_snapshot(session_id)
        base_messages: list[BaseMessage] = []
        after_sequence = 0
        if snapshot:
            try:
                base_messages = messages_from_dict(snapshot["messages"])
                after_sequence = int(snapshot.get("transcript_sequence", 0))
            except Exception as exc:
                logger.warning("会话 {} 的快照反序列化失败: {}", session_id, exc)
                base_messages = []
                after_sequence = 0

        tail_messages, _ = self.load_messages(
            session_id,
            after_sequence=after_sequence,
            include_summaries=snapshot is not None,
        )
        return [*base_messages, *tail_messages]

    def contains_aiops_report(self, session_id: str, normalized_report: str) -> bool:
        messages, _ = self.load_messages(session_id, include_summaries=False)
        for message in messages:
            metadata = getattr(message, "additional_kwargs", {}) or {}
            if (
                isinstance(message, AIMessage)
                and metadata.get("source") == "aiops_report"
                and _message_text(message) == normalized_report
            ):
                return True
        return False

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """返回前端可见的完整历史，不受模型上下文压缩影响。"""
        records = self._read_records(session_id)
        effective_after = self._last_clear_sequence(records)
        history: list[dict[str, str]] = []
        for record in records:
            if record.get("record_type") != "message":
                continue
            if int(record.get("seq", 0)) <= effective_after:
                continue
            serialized = record.get("message")
            if not isinstance(serialized, dict):
                continue
            try:
                message = messages_from_dict([serialized])[0]
            except Exception:
                continue
            if isinstance(message, (SystemMessage, ToolMessage)):
                continue
            metadata = getattr(message, "additional_kwargs", {}) or {}
            if metadata.get("hidden_from_history"):
                continue
            if metadata.get("lc_source") == SUMMARY_SOURCE:
                continue
            content = _message_text(message)
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                if getattr(message, "tool_calls", None) and not content.strip():
                    continue
                role = "assistant"
            else:
                continue
            history.append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": str(record.get("timestamp") or _utc_timestamp()),
                }
            )
        return history
