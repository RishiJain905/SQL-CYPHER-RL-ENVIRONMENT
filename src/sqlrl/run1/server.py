"""Run 1's llama-server: started and stopped from Python (Windows, standard library only).

The server is launched directly (no wrapper script), hidden, with its output in the step's
folder and its pid in a file, so run 1 doesn't share a script, a log folder or a pid file with
anything else. One server at a time: `start` refuses while another llama-server is running.

The flags (measured in `documentation/run1/harness.md`):
- `-ngl 99 -fa on`: all layers and flash attention on the GPU;
- `-np <slots> -c <ctx>` without `-kvu`: every slot has its own KV (`ctx` / `slots` tokens).
  One shared pool (`-kvu`, the pilots' setting) makes each decoding step attend over every
  slot's tokens: 208 tokens/s at 16 slots against 542 with separate KV;
- `--cache-ram 0 -ctxcp 0`: no host-RAM prompt cache (saving a finished request's state stalls
  every slot);
- `--reasoning-budget -1` for rollouts: the raw completion endpoint is used and the thinking
  budget is the harness's own (`rollouts.py`); the evaluation passes the yardstick's 1,024.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from sqlrl.paths import get_paths

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
TEXT = {"text": True, "encoding": "utf-8", "errors": "replace"}
PORT = 8096  # run 1's own port (the pilots used 8095, the trainer 8090)


def running() -> list[str]:
    """Every llama-server process on this machine (tasklist lines)."""
    r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"],
                       capture_output=True, creationflags=NO_WINDOW, **TEXT)  # fmt: skip
    return [line for line in r.stdout.splitlines() if "llama-server" in line]


def healthy(port: int = PORT, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def command(model: Path, port: int, slots: int, ctx: int, lora: Path | None,
            threads: int, extra: list[str] | None, kv_unified: bool = False,
            reasoning_budget: int = -1) -> list[str]:  # fmt: skip
    p = get_paths()
    if p.llama_bin is None:
        raise RuntimeError("configs/paths.local.toml: `llama_bin` is missing")
    args = [str(p.llama_bin / "llama-server.exe"), "-m", str(model), "--host", "127.0.0.1",
            "--port", str(port), "-ngl", "99", "-fa", "on", "-c", str(ctx), "-np", str(slots),
            "-t", str(threads), "--no-webui", "--cache-ram", "0", "-ctxcp", "0",
            "--reasoning-budget", str(reasoning_budget)]  # fmt: skip
    if kv_unified:
        args.append("-kvu")
    if lora is not None:
        args += ["--lora", str(lora)]
    return args + list(extra or [])


def start(model: Path, log_dir: Path, *, port: int = PORT, slots: int = 16, ctx: int = 65536,
          lora: Path | None = None, threads: int = 8, extra: list[str] | None = None,
          kv_unified: bool = False, reasoning_budget: int = -1,
          wait_s: int = 180) -> dict[str, Any]:  # fmt: skip
    """Start the server and wait until it answers; returns {"pid", "seconds", "log"}."""
    if running():
        raise RuntimeError("another llama-server is running: one GPU job at a time")
    p = get_paths()
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if p.rocm_bin is not None:  # the hipBLAS / rocBLAS DLLs
        env["PATH"] = str(p.rocm_bin) + os.pathsep + env.get("PATH", "")
    if p.hip_device is not None:  # the discrete card's index
        env["HIP_VISIBLE_DEVICES"] = str(p.hip_device)
    out, err = log_dir / "server.out.txt", log_dir / "server.err.txt"
    t0 = time.time()
    with out.open("w", encoding="utf-8") as fo, err.open("w", encoding="utf-8") as fe:
        proc = subprocess.Popen(command(model, port, slots, ctx, lora, threads, extra, kv_unified,
                                        reasoning_budget),
                                stdout=fo, stderr=fe, stdin=subprocess.DEVNULL, env=env,
                                creationflags=NO_WINDOW)  # fmt: skip
    (log_dir / "server.pid").write_text(json.dumps({"pid": proc.pid, "port": port}),
                                        encoding="utf-8")  # fmt: skip
    while time.time() - t0 < wait_s:
        if healthy(port):
            return {"pid": proc.pid, "seconds": round(time.time() - t0, 1), "log": str(err)}
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early (code {proc.returncode}); see {err}")
        time.sleep(0.5)
    stop(log_dir)
    raise RuntimeError(f"llama-server not ready after {wait_s} s; see {err}")


def stop(log_dir: Path, wait_s: int = 30) -> bool:
    """Stop the server `start` wrote a pid file for in `log_dir`; True when it is gone."""
    pid_file = log_dir / "server.pid"
    if not pid_file.exists():
        return True
    pid = str(json.loads(pid_file.read_text(encoding="utf-8"))["pid"])
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                       capture_output=True, creationflags=NO_WINDOW, **TEXT)  # fmt: skip
    if "llama-server" in r.stdout:  # never a process that only reuses the pid
        subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True,
                       creationflags=NO_WINDOW, **TEXT)  # fmt: skip
        t0 = time.time()
        while time.time() - t0 < wait_s:
            r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                               capture_output=True, creationflags=NO_WINDOW, **TEXT)  # fmt: skip
            if "llama-server" not in r.stdout:
                break
            time.sleep(0.5)
        else:
            return False
    pid_file.unlink(missing_ok=True)
    return True
