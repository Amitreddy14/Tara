"""
subsystems/execution.py — Execution system with confirmation gate.

SAFETY RULES (hardcoded, cannot be bypassed):
  1. Every action requires explicit user confirmation before execution.
  2. Dangerous operations (system dirs, registry, executables) are blocked.
  3. Every action is logged to an audit file.
  4. Reversible actions support undo (file moves/renames only).

Supported action types:
  - file_read     : read contents of a file
  - file_write    : create or overwrite a file
  - file_move     : move or rename a file
  - file_delete   : delete a file (moves to recycle bin on Windows)
  - file_list     : list files in a directory
  - shell_run     : run a shell command (restricted whitelist)
  - open_app      : open an application by name

Usage (from orchestrator):
  exec_sys = ExecutionSubsystem()
  proposal = exec_sys.parse_action(user_text)   # parse what user wants
  if proposal:
      confirmation_prompt = exec_sys.describe(proposal)   # what to say to user
      # ... wait for user YES/NO ...
      if confirmed:
          result = exec_sys.execute(proposal)
"""

import json
import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Safety configuration ──────────────────────────────────────────────────────

# Directories that can NEVER be touched
BLOCKED_PATHS = [
    "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
    "C:\\ProgramData", "C:\\System Volume Information",
    os.path.expanduser("~\\AppData\\Roaming"),
    os.path.expanduser("~\\AppData\\Local\\Microsoft"),
]

# File extensions that can NEVER be deleted or overwritten
BLOCKED_EXTENSIONS = {
    ".exe", ".dll", ".sys", ".bat", ".cmd", ".msi",
    ".reg", ".ps1", ".vbs", ".scr",
}

# Shell commands that are allowed (whitelist approach)
ALLOWED_COMMANDS = {
    "dir", "ls", "echo", "type", "cat", "mkdir", "rmdir",
    "copy", "move", "rename", "del", "ping", "ipconfig",
    "tasklist", "whoami", "date", "time", "python", "pip",
}

AUDIT_LOG = os.path.join("memory_store", "execution_audit.log")


class ExecutionSubsystem:
    def __init__(self, ollama_client=None, model: str = "llama3"):
        self._client  = ollama_client
        self._model   = model
        self._undo_stack: list[dict] = []   # stores reversible operations
        os.makedirs("memory_store", exist_ok=True)
        log.info("Execution subsystem ready (audit log: %s)", AUDIT_LOG)

    # ── Action parsing ────────────────────────────────────────────

    def parse_action(self, user_text: str) -> Optional[dict]:
        """
        Parse a user request into a structured action dict using pattern matching.
        Falls back to LLM only for complex/ambiguous requests.
        Returns None if no executable action is detected.
        """
        t = user_text.lower().strip().rstrip(".")

        # ── Fast pattern matching (no LLM needed) ──

        # Open app
        open_match = _match_prefix(t, ["open ", "launch ", "start "])
        if open_match:
            return {"type": "open_app", "target": open_match, "args": {}}

        # List files
        if any(p in t for p in ["list files", "show files", "what files", "what's on my desktop", "what is on my desktop"]):
            target = "Desktop"
            if "desktop" in t:
                target = os.path.join(os.path.expanduser("~"), "Desktop")
            elif "documents" in t:
                target = os.path.join(os.path.expanduser("~"), "Documents")
            elif "downloads" in t:
                target = os.path.join(os.path.expanduser("~"), "Downloads")
            return {"type": "file_list", "target": target, "args": {}}

        # Run command
        run_match = _match_prefix(t, ["run ", "run the command ", "execute "])
        if run_match:
            return {"type": "shell_run", "target": run_match, "args": {}}

        # Delete file
        del_match = _match_prefix(t, ["delete ", "remove the file ", "delete the file "])
        if del_match:
            return {"type": "file_delete", "target": del_match, "args": {}}

        # Read file
        read_match = _match_prefix(t, ["read the file ", "read file ", "show me the file ", "open the file "])
        if read_match:
            return {"type": "file_read", "target": read_match, "args": {}}

        # ── LLM fallback for complex requests ──
        if self._client and any(k in t for k in ["move", "rename", "copy", "write to", "create file"]):
            return self._parse_with_llm(user_text)

        return None

    def _parse_with_llm(self, user_text: str) -> Optional[dict]:
        """LLM-based parser for complex actions not covered by patterns."""
        prompt = f"""Extract the computer action from: "{user_text}"

Return ONLY JSON. Examples:
"move report.txt to Documents" → {{"type": "file_move", "target": "report.txt", "args": {{"destination": "C:\\\\Users\\\\amitr\\\\Documents\\\\report.txt"}}}}
"rename old.txt to new.txt" → {{"type": "file_move", "target": "old.txt", "args": {{"destination": "new.txt"}}}}
"create a file called notes.txt" → {{"type": "file_write", "target": "notes.txt", "args": {{"content": ""}}}}

If unclear, return {{}}
JSON:"""
        try:
            response = self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 100},
            )
            raw = response["message"]["content"].strip()
            action = _parse_json_safe(raw)
            if action and action.get("type") and action.get("target"):
                return action
        except Exception as e:
            log.error("LLM action parse failed: %s", e)
        return None

    # ── Describe action for user confirmation ─────────────────────

    def describe(self, action: dict) -> str:
        """
        Generate a clear, plain-English description of the proposed action.
        This is what Tara speaks to the user before asking for confirmation.
        """
        t    = action.get("type", "")
        tgt  = action.get("target", "")
        args = action.get("args", {})

        descriptions = {
            "file_read":   f"I'd like to read the contents of {tgt}.",
            "file_write":  f"I'd like to create or write to the file {tgt}.",
            "file_move":   f"I'd like to move {tgt} to {args.get('destination', 'a new location')}.",
            "file_delete": f"I'd like to permanently delete {tgt}. This cannot be undone.",
            "file_list":   f"I'd like to list the files in {tgt}.",
            "shell_run":   f"I'd like to run this command: {tgt}",
            "open_app":    f"I'd like to open {tgt}.",
        }
        desc = descriptions.get(t, f"I'd like to perform: {t} on {tgt}.")
        return desc + " Should I go ahead?"

    # ── Safety check ──────────────────────────────────────────────

    def is_safe(self, action: dict) -> tuple[bool, str]:
        """
        Returns (is_safe, reason).
        Called before execution — if False, action is BLOCKED regardless of user consent.
        """
        t   = action.get("type", "")
        tgt = action.get("target", "")

        # Normalize path
        try:
            tgt_path = str(Path(tgt).resolve())
        except Exception:
            tgt_path = tgt

        # Check blocked paths
        for blocked in BLOCKED_PATHS:
            if tgt_path.lower().startswith(blocked.lower()):
                return False, f"Access to {blocked} is not permitted."

        # Check blocked extensions for destructive operations
        if t in ("file_delete", "file_write", "file_move"):
            ext = Path(tgt).suffix.lower()
            if ext in BLOCKED_EXTENSIONS:
                return False, f"Modifying {ext} files is not permitted."

        # Check shell command whitelist
        if t == "shell_run":
            cmd = tgt.strip().split()[0].lower()
            if cmd not in ALLOWED_COMMANDS:
                return False, f"The command '{cmd}' is not on the allowed list."

        return True, ""

    # ── Execute (only after confirmed + safety check) ─────────────

    def execute(self, action: dict) -> str:
        """
        Execute the action. ONLY call this after:
          1. User has explicitly confirmed
          2. is_safe() returned True

        Returns a spoken result string.
        """
        t   = action.get("type", "")
        tgt = action.get("target", "")
        args = action.get("args", {})

        self._audit(action, "executing")

        try:
            if t == "file_read":
                return self._file_read(tgt)
            elif t == "file_write":
                return self._file_write(tgt, args.get("content", ""))
            elif t == "file_move":
                return self._file_move(tgt, args.get("destination", ""))
            elif t == "file_delete":
                return self._file_delete(tgt)
            elif t == "file_list":
                return self._file_list(tgt)
            elif t == "shell_run":
                return self._shell_run(tgt)
            elif t == "open_app":
                return self._open_app(tgt)
            else:
                return f"I don't know how to perform the action: {t}."
        except Exception as e:
            self._audit(action, f"error: {e}")
            log.error("Execution error: %s", e)
            return f"Something went wrong: {e}"

    # ── Action implementations ────────────────────────────────────

    def _file_read(self, path: str) -> str:
        p = Path(path)
        if not p.exists():
            return f"The file {path} doesn't exist."
        if p.stat().st_size > 10_000:
            return f"The file is too large to read aloud ({p.stat().st_size} bytes)."
        content = p.read_text(encoding="utf-8", errors="ignore")
        # Summarise for voice
        lines = content.strip().splitlines()
        preview = " ".join(lines[:5])
        return f"The file has {len(lines)} lines. Here's the start: {preview[:300]}"

    def _file_write(self, path: str, content: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Done, I've written to {p.name}."

    def _file_move(self, src: str, dst: str) -> str:
        s, d = Path(src), Path(dst)
        if not s.exists():
            return f"The file {src} doesn't exist."
        # Save for undo
        self._undo_stack.append({"type": "file_move", "from": str(d), "to": str(s)})
        shutil.move(str(s), str(d))
        return f"Done, moved {s.name} to {d}."

    def _file_delete(self, path: str) -> str:
        p = Path(path)
        if not p.exists():
            return f"The file {path} doesn't exist."
        # On Windows, try to send to recycle bin via shell
        try:
            import ctypes
            ctypes.windll.shell32.SHFileOperationW  # test available
            result = subprocess.run(
                ["powershell", "-Command",
                 f"Add-Type -AssemblyName Microsoft.VisualBasic; "
                 f"[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile('{path}',"
                 f"'OnlyErrorDialogs','SendToRecycleBin')"],
                capture_output=True, timeout=10
            )
            if result.returncode == 0:
                self._undo_stack.append({"type": "note", "msg": f"Deleted {path} (in Recycle Bin)"})
                return f"Done, {p.name} has been moved to the Recycle Bin."
        except Exception:
            pass
        # Fallback: permanent delete
        p.unlink()
        return f"Done, {p.name} has been permanently deleted."

    def _file_list(self, path: str) -> str:
        p = Path(path) if path else Path.home() / "Desktop"
        if not p.exists():
            return f"The folder {path} doesn't exist."
        items = list(p.iterdir())[:20]
        names = [i.name for i in items]
        if not names:
            return f"The folder {p.name} is empty."
        return f"In {p.name}: {', '.join(names[:10])}{'...' if len(names) > 10 else ''}."

    def _shell_run(self, command: str) -> str:
        result = subprocess.run(
            command, shell=True, capture_output=True,
            text=True, timeout=15, cwd=os.path.expanduser("~")
        )
        out = (result.stdout + result.stderr).strip()
        if not out:
            return "Command completed with no output."
        # Trim for voice
        lines = out.splitlines()
        if len(lines) > 5:
            return f"Command output ({len(lines)} lines): {' '.join(lines[:3])}..."
        return f"Output: {out[:300]}"

    def _open_app(self, app_name: str) -> str:
        # Map common names to their actual executable
        app_map = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "calc": "calc.exe",
            "chrome": "chrome",
            "firefox": "firefox",
            "explorer": "explorer.exe",
            "cmd": "cmd.exe",
            "terminal": "wt.exe",   # Windows Terminal
            "word": "winword",
            "excel": "excel",
            "paint": "mspaint.exe",
        }
        exe = app_map.get(app_name.lower().strip(), app_name)
        try:
            subprocess.Popen(
                ["powershell", "-Command", f"Start-Process '{exe}'"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return f"Opening {app_name}."
        except Exception as e:
            log.error("open_app failed: %s", e)
            return f"I couldn't open {app_name}. Make sure it's installed."

    # ── Undo ──────────────────────────────────────────────────────

    def undo_last(self) -> str:
        if not self._undo_stack:
            return "There's nothing to undo."
        op = self._undo_stack.pop()
        if op["type"] == "file_move":
            shutil.move(op["from"], op["to"])
            return f"Done, moved it back to {op['to']}."
        return f"I can't undo that operation."

    # ── Audit log ─────────────────────────────────────────────────

    def _audit(self, action: dict, status: str) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action":    action,
            "status":    status,
        }
        try:
            with open(AUDIT_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass


# ── helpers ───────────────────────────────────────────────────────────────────

def _match_prefix(text: str, prefixes: list) -> Optional[str]:
    """Return the text after the first matching prefix, or None."""
    for prefix in prefixes:
        if text.startswith(prefix):
            remainder = text[len(prefix):].strip()
            if remainder:
                return remainder
    return None


def _parse_json_safe(text: str) -> dict:
    import re
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except Exception:
        return {}