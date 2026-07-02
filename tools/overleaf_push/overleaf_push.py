#!/usr/bin/env python3
"""Global Overleaf helper CLI."""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import logging
import mimetypes
import os
import posixpath
import re
import time
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

import requests


LOG = logging.getLogger("overleaf_push")
GLOBAL_CONFIG_DIR = Path.home() / ".overleaf-helper"
GLOBAL_CONFIG_PATH = GLOBAL_CONFIG_DIR / "config.json"
MAPPING_FILENAME = ".overleaf"
COOKIE_ATTRIBUTE_NAMES = {
    "domain",
    "expires",
    "path",
    "samesite",
    "secure",
    "httponly",
    "max-age",
}


@dataclass
class AccountConfig:
    server_url: str
    cookie: str
    project_id: str
    project_name: str


@dataclass
class SyncRule:
    source: str
    target_dir: str
    exclude: List[str] = field(default_factory=list)


@dataclass
class ResolvedRule:
    rule: SyncRule
    entries: List[Tuple[Path, str]]
    scope_prefix: str


@dataclass
class GeneratedRule:
    source: str
    target_dir: str
    remote_path: str
    local_path: str


@dataclass
class SyncReport:
    changed: List[str] = field(default_factory=list)
    uploaded: List[str] = field(default_factory=list)
    missing_local: List[str] = field(default_factory=list)
    same: List[str] = field(default_factory=list)


class OverleafClient:
    def __init__(self, server_url: str, cookie: str):
        parsed = urlsplit(server_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"invalid server_url: {server_url}")
        self.base_url = f"{parsed.scheme}://{parsed.netloc}/"
        self.origin = f"{parsed.scheme}://{parsed.netloc}"
        self.host = parsed.netloc
        self.cookie = cookie.strip()
        self.session = requests.Session()
        self.csrf_token = ""
        self.user_id = ""
        self.user_email = ""
        self.current_project_id = ""
        self._seed_session_cookies()

    def _seed_session_cookies(self) -> None:
        host = urlsplit(self.base_url).hostname or ""
        domain = "." + host.removeprefix("www.")
        for part in self.cookie.split(";"):
            if "=" not in part:
                continue
            name, value = part.strip().split("=", 1)
            if name.lower() in COOKIE_ATTRIBUTE_NAMES:
                continue
            self.session.cookies.set(name, value, domain=domain)

    def _headers(self, extra: Optional[Dict[str, str]] = None, *, include_origin: bool = False) -> Dict[str, str]:
        headers = {
            "Connection": "keep-alive",
        }
        if include_origin:
            headers["Origin"] = self.origin
        if extra:
            headers.update(extra)
        return headers

    def login(self) -> None:
        res = self.session.get(self.base_url + "project", headers=self._headers(), allow_redirects=True, timeout=30)
        res.raise_for_status()
        body = res.text

        def meta(name: str) -> str:
            match = re.search(rf'<meta\s+name="{re.escape(name)}"\s+content="([^"]*)">', body)
            if not match:
                raise RuntimeError(f"missing meta tag: {name}")
            return unescape(match.group(1))

        self.user_id = meta("ol-user_id")
        self.user_email = meta("ol-usersEmail")
        self.csrf_token = meta("ol-csrfToken")

        js_res = self.session.get(self.base_url + "socket.io/socket.io.js", headers=self._headers(), timeout=30)
        js_res.raise_for_status()

    def list_projects(self) -> List[Dict[str, Any]]:
        res = self.session.post(
            self.base_url + "api/project",
            json={"_csrf": self.csrf_token},
            headers=self._headers({"X-Csrf-Token": self.csrf_token}),
            timeout=30,
        )
        res.raise_for_status()
        payload = res.json()
        return payload.get("projects", [])

    def find_project(self, project_id: Optional[str], project_name: Optional[str]) -> Dict[str, Any]:
        projects = self.list_projects()
        if project_id:
            for project in projects:
                if project.get("id") == project_id or project.get("_id") == project_id:
                    return project
            raise RuntimeError(f"project_id not found: {project_id}")

        matches = [p for p in projects if p.get("name") == project_name]
        if not matches:
            raise RuntimeError(f"project_name not found: {project_name}")
        if len(matches) > 1:
            raise RuntimeError(f"project_name is ambiguous: {project_name}")
        return matches[0]

    @staticmethod
    def _decode_socketio_payload(data: str) -> List[str]:
        if not data:
            return []
        if not data.startswith("\ufffd"):
            return [data]
        packets: List[str] = []
        idx = 0
        while idx < len(data):
            if data[idx] != "\ufffd":
                raise RuntimeError(f"unexpected socket.io payload framing: {data[:100]!r}")
            next_sep = data.find("\ufffd", idx + 1)
            if next_sep < 0:
                raise RuntimeError("malformed socket.io payload length framing")
            size = int(data[idx + 1:next_sep])
            start = next_sep + 1
            packets.append(data[start:start + size])
            idx = start + size
        return packets

    @staticmethod
    def _decode_socketio_packet(packet: str) -> Dict[str, Any]:
        match = re.match(r"([^:]+):([0-9]+)?(\+)?:([^:]+)?:?([\s\S]*)?", packet)
        if not match:
            return {"raw": packet, "type": "unknown"}
        code, packet_id, ack_plus, endpoint, tail = match.groups()
        tail = tail or ""
        parsed: Dict[str, Any] = {
            "raw": packet,
            "code": int(code),
            "id": packet_id,
            "ack": bool(ack_plus),
            "endpoint": endpoint or "",
            "tail": tail,
        }
        if parsed["code"] == 1:
            parsed["type"] = "connect"
        elif parsed["code"] == 5:
            event = json.loads(tail)
            parsed["type"] = "event"
            parsed["name"] = event.get("name")
            parsed["args"] = event.get("args", [])
        elif parsed["code"] == 6:
            ack_match = re.match(r"^([0-9]+)(\+)?([\s\S]*)", tail)
            if not ack_match:
                raise RuntimeError(f"malformed socket.io ack packet: {packet!r}")
            parsed["type"] = "ack"
            parsed["ackId"] = ack_match.group(1)
            ack_tail = ack_match.group(3)
            parsed["args"] = json.loads(ack_tail) if ack_tail else []
        elif parsed["code"] == 7:
            reasons = ["transport not supported", "client not handshaken", "unauthorized"]
            advice = ["reconnect"]
            reason_idx, _, advice_idx = tail.partition("+")
            parsed["type"] = "error"
            parsed["reason"] = reasons[int(reason_idx)] if reason_idx else ""
            parsed["advice"] = advice[int(advice_idx)] if advice_idx else ""
        else:
            parsed["type"] = "other"
        return parsed

    def _socketio_open(self, project_id: str) -> str:
        res = self.session.get(
            self.base_url + f"socket.io/1/?projectId={project_id}&t={int(time.time() * 1000)}",
            headers=self._headers(include_origin=True),
            timeout=30,
        )
        res.raise_for_status()
        return res.text.split(":", 1)[0]

    def _socketio_base(self, project_id: str, session_id: str) -> str:
        return self.base_url + f"socket.io/1/xhr-polling/{session_id}?projectId={project_id}"

    def _socketio_poll(self, base_url: str) -> List[Dict[str, Any]]:
        res = self.session.get(
            base_url + f"&t={int(time.time() * 1000)}",
            headers=self._headers(include_origin=True),
            timeout=30,
        )
        res.raise_for_status()
        packets = [self._decode_socketio_packet(packet) for packet in self._decode_socketio_payload(res.text)]
        for packet in packets:
            if packet.get("type") == "error":
                raise RuntimeError(packet.get("reason") or packet["raw"])
        return packets

    def _socketio_post(self, base_url: str, payload: str) -> None:
        res = self.session.post(
            base_url + f"&t={int(time.time() * 1000)}",
            data=payload.encode("utf-8"),
            headers=self._headers({"Content-Type": "text/plain;charset=UTF-8"}, include_origin=True),
            timeout=30,
        )
        res.raise_for_status()

    def _socketio_request_event(self, project_id: str, event_name: str, args: List[Any], response_event: str) -> List[Any]:
        session_id = self._socketio_open(project_id)
        base_url = self._socketio_base(project_id, session_id)
        self._socketio_poll(base_url)  # initial connect packet
        payload = "5:::" + json.dumps({"name": event_name, "args": args}, separators=(",", ":"))
        self._socketio_post(base_url, payload)
        deadline = time.time() + 30
        while time.time() < deadline:
            packets = self._socketio_poll(base_url)
            for packet in packets:
                if packet.get("type") == "event" and packet.get("name") == response_event:
                    return packet.get("args", [])
        raise RuntimeError(f"timed out waiting for socket.io event {response_event}")

    def _socketio_request_ack(self, project_id: str, event_name: str, args: List[Any]) -> List[Any]:
        session_id = self._socketio_open(project_id)
        base_url = self._socketio_base(project_id, session_id)
        self._socketio_poll(base_url)  # initial connect packet
        payload = "5:1+::" + json.dumps({"name": event_name, "args": args}, separators=(",", ":"))
        self._socketio_post(base_url, payload)
        deadline = time.time() + 30
        while time.time() < deadline:
            packets = self._socketio_poll(base_url)
            for packet in packets:
                if packet.get("type") == "ack" and packet.get("ackId") == "1":
                    return packet.get("args", [])
        raise RuntimeError(f"timed out waiting for socket.io ack {event_name}")

    def join_project(self, project_id: str) -> Dict[str, Any]:
        args = self._socketio_request_event(project_id, "joinProject", [{"project_id": project_id}], "joinProjectResponse")
        if not args:
            raise RuntimeError("joinProjectResponse did not contain payload")
        return args[0]["project"]

    def join_doc(self, doc_id: str) -> str:
        project_id = self.current_project_id
        args = self._socketio_request_ack(project_id, "joinDoc", [doc_id, {"encodeRanges": True}])
        if len(args) < 2:
            raise RuntimeError(f"unexpected joinDoc ack payload: {args!r}")
        doc_lines = args[0]
        if isinstance(doc_lines, list):
            return "\n".join(doc_lines)
        raise RuntimeError(f"unexpected joinDoc lines payload: {type(doc_lines)!r}")

    def add_folder(self, project_id: str, parent_folder_id: str, folder_name: str) -> Dict[str, Any]:
        res = self.session.post(
            self.base_url + f"project/{project_id}/folder",
            json={"_csrf": self.csrf_token, "name": folder_name, "parent_folder_id": parent_folder_id},
            headers=self._headers({"X-Csrf-Token": self.csrf_token}),
            timeout=30,
        )
        res.raise_for_status()
        return res.json()

    def add_doc(self, project_id: str, parent_folder_id: str, filename: str) -> Dict[str, Any]:
        res = self.session.post(
            self.base_url + f"project/{project_id}/doc",
            json={"_csrf": self.csrf_token, "parent_folder_id": parent_folder_id, "name": filename},
            headers=self._headers({"X-Csrf-Token": self.csrf_token}),
            timeout=30,
        )
        res.raise_for_status()
        return res.json()

    def upload_file(self, project_id: str, parent_folder_id: str, filename: str, content: bytes) -> Dict[str, Any]:
        mime_type = mimetypes.guess_type(filename)[0] or "text/plain"
        res = self.session.post(
            self.base_url + f"project/{project_id}/upload?folder_id={parent_folder_id}",
            data={
                "targetFolderId": parent_folder_id,
                "name": filename,
                "type": mime_type,
            },
            files={"qqfile": (filename, content)},
            headers=self._headers({"X-Csrf-Token": self.csrf_token}),
            timeout=120,
        )
        res.raise_for_status()
        return res.json()

    def delete_entity(self, project_id: str, entity_type: str, entity_id: str) -> None:
        res = self.session.delete(
            self.base_url + f"project/{project_id}/{entity_type}/{entity_id}",
            headers=self._headers({"X-Csrf-Token": self.csrf_token}),
            timeout=30,
        )
        res.raise_for_status()

    def download_file(self, project_id: str, file_id: str) -> bytes:
        res = self.session.get(
            self.base_url + f"project/{project_id}/file/{file_id}",
            headers=self._headers(),
            timeout=120,
        )
        res.raise_for_status()
        return res.content


def normalize_remote_path(path: str) -> str:
    path = path.replace("\\", "/").strip()
    if not path or path == ".":
        return ""
    path = posixpath.normpath(path)
    if path in (".", "/"):
        return ""
    return path.strip("/")


def join_remote(*parts: str) -> str:
    cleaned = [normalize_remote_path(part) for part in parts if normalize_remote_path(part)]
    return "/".join(cleaned)


def match_patterns(rel_path: str, include: Sequence[str], exclude: Sequence[str]) -> bool:
    rel_path = rel_path.replace("\\", "/")
    if include and not any(fnmatch.fnmatchcase(rel_path, pat.replace("\\", "/")) for pat in include):
        return False
    if exclude and any(fnmatch.fnmatchcase(rel_path, pat.replace("\\", "/")) for pat in exclude):
        return False
    return True


def iter_local_files(root: Path) -> Iterator[Path]:
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in files:
            yield Path(current_root) / name


def iter_local_files_by_suffix(root: Path, suffix: str) -> Iterator[Path]:
    suffix = suffix.lower()
    for path in iter_local_files(root):
        if path.suffix.lower() == suffix:
            yield path


def select_pdf_match(repo_root: Path, remote_path: str, matches: Sequence[Path]) -> Path:
    remote_parent_parts = [part for part in normalize_remote_path(posixpath.dirname(remote_path)).split("/") if part]
    remote_name = posixpath.basename(remote_path).lower()

    def score(path: Path) -> Tuple[int, int, int, str]:
        rel_parts = path.relative_to(repo_root).parts
        rel_dir_parts = [part.lower() for part in rel_parts[:-1]]
        archive_penalty = 1 if any(part.lower() == "expr_pdf_archive" for part in rel_parts) else 0
        overlap = len(set(remote_parent_parts).intersection(rel_dir_parts))
        name_hint = 1 if path.name.lower() == remote_name else 0
        depth = len(rel_parts)
        return (archive_penalty, -overlap, -name_hint, depth, path.as_posix())

    ranked = sorted(matches, key=score)
    return ranked[0]


def find_repo_root(start: Path) -> Optional[Path]:
    current = start.resolve()
    if current.is_file():
        current = current.parent

    while True:
        if (current / MAPPING_FILENAME).is_file():
            return current
        if current.parent == current:
            return None
        current = current.parent


def ensure_repo_root(start: Path) -> Path:
    root = find_repo_root(start)
    if root is None:
        raise RuntimeError(
            f'No {MAPPING_FILENAME} found from "{start}". '
            f'Run "overleaf-helper --init" in the directory you want to use as the repo base.'
        )
    return root


def has_glob_magic(text: str) -> bool:
    return any(ch in text for ch in "*?[")


def glob_base(pattern: str) -> Path:
    normalized = pattern.replace("\\", "/")
    parts = normalized.split("/")
    base_parts: List[str] = []
    for part in parts:
        if has_glob_magic(part):
            break
        base_parts.append(part)
    return Path("/".join(base_parts)) if base_parts else Path(".")


def resolve_local_path(repo_root: Path, text: str) -> Path:
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_cookie_header(cookie: str) -> str:
    parts: List[str] = []
    for part in cookie.split(";"):
        if "=" not in part:
            continue
        key, value = part.strip().split("=", 1)
        if key.lower() in COOKIE_ATTRIBUTE_NAMES:
            continue
        parts.append(f"{key}={value}")
    return "; ".join(parts)


def load_mapping_file(repo_root: Path) -> List[SyncRule]:
    mapping_path = repo_root / MAPPING_FILENAME
    if not mapping_path.exists():
        raise RuntimeError(f'Missing {MAPPING_FILENAME} in {repo_root}')

    rules: List[SyncRule] = []
    for lineno, raw_line in enumerate(mapping_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "->" not in line:
            raise RuntimeError(f'{mapping_path}:{lineno}: missing "->" separator')
        source, target_dir = line.split("->", 1)
        source = source.strip()
        target_dir = target_dir.strip()
        if not source:
            raise RuntimeError(f"{mapping_path}:{lineno}: empty source path")
        rules.append(SyncRule(source=source, target_dir=target_dir, exclude=[]))
    return rules


def write_mapping_file(repo_root: Path, rules: Sequence[SyncRule]) -> Path:
    mapping_path = repo_root / MAPPING_FILENAME
    mapping_path.write_text(
        "\n".join(f"{rule.source} -> {rule.target_dir}" for rule in rules) + ("\n" if rules else ""),
        encoding="utf-8",
    )
    return mapping_path


def init_repo(repo_root: Path) -> Path:
    mapping_path = repo_root / MAPPING_FILENAME
    if mapping_path.exists():
        raise RuntimeError(f"{mapping_path} already exists")
    mapping_path.write_text("# source_path -> overleaf_dir\n", encoding="utf-8")
    return mapping_path


def prompt_text(label: str, default: Optional[str] = None, secret: bool = False) -> str:
    prompt = f"{label}"
    if default:
        prompt += f" [{default}]"
    prompt += ": "
    if secret:
        import getpass

        value = getpass.getpass(prompt)
    else:
        value = input(prompt)
    value = value.strip()
    return value or (default or "")


def prompt_choice(label: str, items: Sequence[str]) -> int:
    print(label)
    for idx, item in enumerate(items, start=1):
        print(f"  {idx}. {item}")
    while True:
        raw = input("Select number: ").strip()
        try:
            idx = int(raw)
        except ValueError:
            continue
        if 1 <= idx <= len(items):
            return idx - 1


def load_global_config(path: Path, bootstrap: bool) -> AccountConfig:
    raw = load_json(path)
    changed = False

    server_url = raw.get("server_url") or prompt_text("Overleaf server URL", "https://www.overleaf.com")
    cookie = normalize_cookie_header(raw.get("cookie") or prompt_text("Overleaf cookie", secret=True))
    project_id = raw.get("project_id")
    project_name = raw.get("project_name")

    client = OverleafClient(server_url, cookie)
    while True:
        try:
            client.login()
            break
        except Exception as exc:
            LOG.warning("Login failed: %s", exc)
            cookie = normalize_cookie_header(prompt_text("Overleaf cookie", secret=True))
            changed = True
            client = OverleafClient(server_url, cookie)

    if bootstrap or not project_id or not project_name:
        projects = client.list_projects()
        if not projects:
            raise RuntimeError("no projects found for this account")
        labels = [f"{p.get('name')} ({p.get('id') or p.get('_id')})" for p in projects]
        selected = prompt_choice("Select target paper/project:", labels)
        project = projects[selected]
        project_id = project.get("id") or project.get("_id")
        project_name = project.get("name")
        changed = True
    else:
        try:
            project = client.find_project(project_id, project_name)
            project_id = project.get("id") or project.get("_id")
            project_name = project.get("name")
        except Exception:
            projects = client.list_projects()
            labels = [f"{p.get('name')} ({p.get('id') or p.get('_id')})" for p in projects]
            selected = prompt_choice("Saved project not found, pick again:", labels)
            project = projects[selected]
            project_id = project.get("id") or project.get("_id")
            project_name = project.get("name")
            changed = True

    account = AccountConfig(
        server_url=server_url,
        cookie=cookie,
        project_id=str(project_id),
        project_name=str(project_name),
    )
    if changed or not path.exists():
        save_json(
            path,
            {
                "server_url": account.server_url,
                "cookie": account.cookie,
                "project_id": account.project_id,
                "project_name": account.project_name,
            },
        )
        LOG.warning("Saved account config to %s", path)
    return account


def expand_rule(rule: SyncRule, repo_root: Path) -> ResolvedRule:
    source_text = rule.source
    target_dir = normalize_remote_path(rule.target_dir)
    entries: List[Tuple[Path, str]] = []

    if has_glob_magic(source_text):
        pattern_path = str((repo_root / source_text).resolve())
        base_dir = resolve_local_path(repo_root, str(glob_base(source_text)))
        matches = sorted(Path(p) for p in glob.glob(pattern_path, recursive=True))
        for local_path in matches:
            if not local_path.is_file():
                continue
            try:
                rel = local_path.relative_to(base_dir).as_posix()
            except ValueError:
                rel = local_path.name
            if not match_patterns(rel, [], rule.exclude):
                continue
            entries.append((local_path, join_remote(target_dir, rel)))
        return ResolvedRule(rule=rule, entries=entries, scope_prefix=target_dir)

    source_path = resolve_local_path(repo_root, source_text)
    if source_path.is_dir():
        for local_path in iter_local_files(source_path):
            rel = local_path.relative_to(source_path).as_posix()
            if not match_patterns(rel, [], rule.exclude):
                continue
            entries.append((local_path, join_remote(target_dir, rel)))
        return ResolvedRule(rule=rule, entries=entries, scope_prefix=target_dir)

    if source_path.is_file():
        rel = source_path.name
        if not match_patterns(rel, [], rule.exclude):
            return ResolvedRule(rule=rule, entries=[], scope_prefix=target_dir)
        entries.append((source_path, join_remote(target_dir, rel)))
        return ResolvedRule(rule=rule, entries=entries, scope_prefix=target_dir)

    return ResolvedRule(rule=rule, entries=[], scope_prefix=target_dir)


def build_remote_tree(project: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    root_folders = project.get("rootFolder") or []
    if not root_folders:
        raise RuntimeError("project tree does not contain rootFolder")

    folders: Dict[str, Dict[str, Any]] = {}
    files: Dict[str, Dict[str, Any]] = {}

    def walk(folder: Dict[str, Any], rel_dir: str) -> None:
        folders[rel_dir] = folder
        for child in folder.get("folders", []) or []:
            walk(child, join_remote(rel_dir, child.get("name", "")))
        for entity in folder.get("docs", []) or []:
            files[join_remote(rel_dir, entity.get("name", ""))] = {"kind": "doc", "entity": entity}
        for entity in folder.get("fileRefs", []) or []:
            files[join_remote(rel_dir, entity.get("name", ""))] = {"kind": "file", "entity": entity}

    walk(root_folders[0], "")
    return folders, files


def autogen_rules(account: AccountConfig, repo_root: Path) -> List[SyncRule]:
    client = OverleafClient(account.server_url, account.cookie)
    client.login()
    project = client.find_project(account.project_id, account.project_name)
    project_id = project.get("id") or project.get("_id")
    if not project_id:
        raise RuntimeError("project id is missing from api/project response")
    client.current_project_id = str(project_id)

    LOG.info("Scanning project for pdf files: %s (%s)", account.project_name, project_id)
    remote_project = client.join_project(str(project_id))
    _, remote_files = build_remote_tree(remote_project)

    remote_pdfs = sorted(
        remote_path for remote_path in remote_files.keys()
        if remote_path.lower().endswith(".pdf")
    )
    if not remote_pdfs:
        raise RuntimeError("no remote pdf files found in selected Overleaf project")

    local_by_name: Dict[str, List[Path]] = {}
    for local_path in iter_local_files_by_suffix(repo_root, ".pdf"):
        local_by_name.setdefault(local_path.name, []).append(local_path)

    generated: List[GeneratedRule] = []
    skipped_missing: List[str] = []
    resolved_ambiguous: List[Tuple[str, Path, List[Path]]] = []

    for remote_path in remote_pdfs:
        basename = posixpath.basename(remote_path)
        matches = sorted(local_by_name.get(basename, []))
        if not matches:
            skipped_missing.append(remote_path)
            continue
        local_path = select_pdf_match(repo_root, remote_path, matches)
        generated.append(
            GeneratedRule(
                source=local_path.relative_to(repo_root).as_posix(),
                target_dir=normalize_remote_path(posixpath.dirname(remote_path)),
                remote_path=remote_path,
                local_path=local_path.as_posix(),
            )
        )
        if len(matches) > 1:
            resolved_ambiguous.append((remote_path, local_path, matches))

    rules = [SyncRule(source=item.source, target_dir=item.target_dir, exclude=[]) for item in generated]
    write_mapping_file(repo_root, rules)

    LOG.warning("Autogenerated %d rules into %s", len(rules), repo_root / MAPPING_FILENAME)
    for item in generated:
        LOG.info("[autogen] %s -> %s", item.source, item.remote_path)
    for remote_path in skipped_missing:
        LOG.warning("[autogen-skip-missing] no local pdf named %s for remote %s", posixpath.basename(remote_path), remote_path)
    for remote_path, chosen, matches in resolved_ambiguous:
        rendered = ", ".join(match.relative_to(repo_root).as_posix() for match in matches)
        LOG.warning("[autogen-heuristic] chose %s for remote %s among: %s", chosen.relative_to(repo_root).as_posix(), remote_path, rendered)

    return rules


def resolve_remote_folder_id(
    client: OverleafClient,
    project_id: str,
    folders: Dict[str, Dict[str, Any]],
    remote_dir: str,
) -> str:
    remote_dir = normalize_remote_path(remote_dir)
    if remote_dir in folders:
        return folders[remote_dir]["_id"]
    if not remote_dir:
        raise RuntimeError("missing root folder")

    parent_dir = normalize_remote_path(posixpath.dirname(remote_dir))
    folder_name = posixpath.basename(remote_dir)
    parent_id = resolve_remote_folder_id(client, project_id, folders, parent_dir)
    folder = client.add_folder(project_id, parent_id, folder_name)
    folders[remote_dir] = folder
    return folder["_id"]


def remote_entry_bytes(client: OverleafClient, project_id: str, entry: Dict[str, Any]) -> bytes:
    kind = entry["kind"]
    entity = entry["entity"]
    if kind == "file":
        return client.download_file(project_id, entity["_id"])
    if kind == "doc":
        content = client.join_doc(entity["_id"])
        return content.encode("utf-8")
    raise RuntimeError(f"unsupported remote entry kind: {kind}")


def sync_entry(
    client: OverleafClient,
    project_id: str,
    folders: Dict[str, Dict[str, Any]],
    remote_files: Dict[str, Dict[str, Any]],
    local_path: Path,
    remote_path: str,
    dry_run: bool,
    report: SyncReport,
) -> None:
    local_bytes = local_path.read_bytes()
    remote_entry = remote_files.get(remote_path)
    remote_parent = normalize_remote_path(posixpath.dirname(remote_path))
    remote_name = posixpath.basename(remote_path)

    if remote_entry is None:
        report.uploaded.append(remote_path)
        if dry_run:
            LOG.info("[new] %s -> %s", local_path, remote_path)
            return
        parent_id = resolve_remote_folder_id(client, project_id, folders, remote_parent)
        if len(local_bytes) == 0:
            client.add_doc(project_id, parent_id, remote_name)
        else:
            client.upload_file(project_id, parent_id, remote_name, local_bytes)
        LOG.info("[upload] %s -> %s", local_path, remote_path)
        return

    remote_bytes = remote_entry_bytes(client, project_id, remote_entry)
    if remote_bytes == local_bytes:
        report.same.append(remote_path)
        LOG.info("[same] %s", remote_path)
        return

    if remote_entry["kind"] != "file":
        LOG.warning("Skip replace for remote doc %s; only file replacement is implemented.", remote_path)
        return

    report.changed.append(remote_path)
    if dry_run:
        LOG.info("[replace] %s -> %s", local_path, remote_path)
        return

    parent_id = resolve_remote_folder_id(client, project_id, folders, remote_parent)
    client.upload_file(project_id, parent_id, remote_name, local_bytes)
    LOG.info("[replace] %s -> %s", local_path, remote_path)


def print_sync_report(report: SyncReport) -> None:
    LOG.warning(
        "Summary: changed=%d uploaded=%d missing_local=%d same=%d",
        len(report.changed),
        len(report.uploaded),
        len(report.missing_local),
        len(report.same),
    )
    for remote_path in report.changed:
        LOG.warning("[summary-changed] %s", remote_path)
    for remote_path in report.uploaded:
        LOG.warning("[summary-uploaded] %s", remote_path)
    for detail in report.missing_local:
        LOG.warning("[summary-missing-local] %s", detail)


def warn_unmanaged_remote_files(
    remote_files: Dict[str, Dict[str, Any]],
    managed_scopes: Sequence[str],
    expected_paths: Sequence[str],
) -> None:
    expected = set(expected_paths)
    warned: set[str] = set()
    for remote_path in sorted(remote_files.keys()):
        for scope in managed_scopes:
            scope = normalize_remote_path(scope)
            if scope:
                if not (remote_path == scope or remote_path.startswith(scope + "/")):
                    continue
            if remote_path not in expected and remote_path not in warned:
                LOG.warning("Remote file exists but is unmanaged: %s", remote_path)
                warned.add(remote_path)


def run_sync(account: AccountConfig, repo_root: Path, rules: List[SyncRule], dry_run: bool) -> None:
    client = OverleafClient(account.server_url, account.cookie)
    client.login()
    project = client.find_project(account.project_id, account.project_name)
    project_id = project.get("id") or project.get("_id")
    if not project_id:
        raise RuntimeError("project id is missing from api/project response")
    client.current_project_id = str(project_id)

    LOG.info("Using project: %s (%s)", account.project_name, project_id)
    remote_project = client.join_project(str(project_id))
    folders, remote_files = build_remote_tree(remote_project)

    resolved_rules = [expand_rule(rule, repo_root) for rule in rules]
    planned: Dict[str, Path] = {}
    managed_scopes: List[str] = []
    report = SyncReport()

    for resolved in resolved_rules:
        managed_scopes.append(resolved.scope_prefix)
        if not resolved.entries:
            report.missing_local.append(resolved.rule.source)
            LOG.warning("[missing-local] no local matches for source: %s", resolved.rule.source)
            continue
        for local_path, remote_path in resolved.entries:
            previous = planned.get(remote_path)
            if previous is not None and previous != local_path:
                raise RuntimeError(f"remote path mapped by multiple sources: {remote_path}")
            planned[remote_path] = local_path

    for remote_path, local_path in sorted(planned.items()):
        sync_entry(client, str(project_id), folders, remote_files, local_path, remote_path, dry_run, report)

    warn_unmanaged_remote_files(remote_files, managed_scopes, list(planned.keys()))
    print_sync_report(report)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Push local files into Overleaf.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--sync", action="store_true", help="Sync files according to .overleaf.")
    action.add_argument("--autogen-rules", action="store_true", help="Generate .overleaf rules from matching PDF names.")
    action.add_argument("--init", action="store_true", help="Create an empty .overleaf in the current directory.")
    parser.add_argument("--bootstrap-account", action="store_true", help="Recreate ~/.overleaf-helper/config.json interactively.")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without uploading.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")

    if args.init:
        mapping_path = init_repo(Path.cwd())
        LOG.warning("Created %s", mapping_path)
        return 0

    account = load_global_config(GLOBAL_CONFIG_PATH, args.bootstrap_account)
    repo_root = ensure_repo_root(Path.cwd())

    if args.autogen_rules:
        autogen_rules(account, repo_root)
        return 0

    rules = load_mapping_file(repo_root)
    run_sync(account, repo_root, rules, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
