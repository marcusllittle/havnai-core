"""Free the 12 GB card before a Wan or LTX-2.3 job, then put ACE-Step back.

Z-Image stays loaded between image jobs and is about 10 GB. ACE-Step 1.5 is a
separate process (``acestep-api`` on port 8001) and keeps its weights on the
card. It has no unload route. A Wan clip peaks near 11 GB and LTX-2.3 near
4.7 GB, so neither fits beside ACE-Step or Z-Image.

A video job stops ACE-Step before it takes the GPU, waits until the memory is
actually free, and starts the service again when the job ends. A music job
waits while that video job holds the card, or drops Z-Image and starts
ACE-Step when the card is free. A failed handoff does not try again until a
short cooldown has passed.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional


class GpuBusy(RuntimeError):
    """The card is still occupied, so this job must not start."""


LogFn = Callable[[str], None]
CommandFn = Callable[[str], Any]

DEFAULT_ACE_STOP = "systemctl --user stop acestep-api"
DEFAULT_ACE_START = "systemctl --user start acestep-api"
IDLE_MIB = 2048
VRAM_WAIT_SECONDS = 20.0
HANDOFF_COOLDOWN_SECONDS = 60.0
_UNLOAD_PATHS = ("/v1/unload_model", "/v1/models/unload", "/unload_model")

_cooldown_until = 0.0
_gpu_lock = threading.Lock()


@dataclass
class CommandResult:
    returncode: int
    stdout: str = ""


@dataclass
class HandoffState:
    zimage_was_loaded: bool = False
    ace_up: bool = False
    ace_model: str = ""
    ace_stopped: bool = False
    ace_unloaded: bool = False
    holds_lock: bool = False


def _quiet(message: str) -> None:
    return None


def _zimage_module() -> Any:
    try:
        from client import zimage_turbo
    except ImportError:
        import zimage_turbo  # type: ignore
    return zimage_turbo


def _ace_base() -> str:
    return os.environ.get("ACESTEP_API_URL", "http://127.0.0.1:8001").rstrip("/")


def _stop_command() -> str:
    return os.environ.get("HAVNAI_ACESTEP_STOP_CMD", "").strip() or DEFAULT_ACE_STOP


def _start_command() -> str:
    return os.environ.get("HAVNAI_ACESTEP_START_CMD", "").strip() or DEFAULT_ACE_START


def _returncode(result: Any) -> int:
    if isinstance(result, int):
        return result
    return int(getattr(result, "returncode", 1))


def _stdout(result: Any) -> str:
    if isinstance(result, int):
        return ""
    return str(getattr(result, "stdout", "") or "")


def _run_command(command: str) -> CommandResult:
    try:
        completed = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return CommandResult(1, "")
    return CommandResult(int(completed.returncode), completed.stdout or "")


def _release_lock(state: HandoffState) -> None:
    if not state.holds_lock:
        return
    state.holds_lock = False
    try:
        _gpu_lock.release()
    except RuntimeError:
        return None


def ace_status(session: Any, base_url: str) -> dict:
    """Return ``{up, loaded_model}``. A refused connection means it is down."""
    try:
        response = session.get(f"{base_url}/health", timeout=3)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return {"up": False, "loaded_model": ""}
    data = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload
    if not isinstance(data, dict):
        data = {}
    loaded = str(data.get("loaded_model") or "").strip()
    if data.get("models_initialized") is False:
        loaded = ""
    return {"up": True, "loaded_model": loaded}


def ace_http_unload(session: Any, base_url: str) -> bool:
    paths = [os.environ.get("HAVNAI_ACESTEP_UNLOAD_PATH", "").strip(), *_UNLOAD_PATHS]
    for path in paths:
        if not path:
            continue
        try:
            response = session.post(f"{base_url}/{path.lstrip('/')}", json={}, timeout=20)
        except Exception:
            continue
        if response.status_code < 400:
            return True
    return False


def _read_vram_mib(run_command: CommandFn) -> Optional[int]:
    try:
        result = run_command("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
    except Exception:
        return None
    if _returncode(result) != 0:
        return None
    line = _stdout(result).strip().splitlines()
    if not line:
        return None
    token = line[0].split(",")[0].strip()
    try:
        return int(float(token))
    except ValueError:
        return None


def _memory_released(used: Optional[int]) -> bool:
    """Missing readings are treated as unknown, which must not block a machine without nvidia-smi."""
    return used is None or used <= IDLE_MIB


def _wait_until_clear(
    session: Any,
    base_url: str,
    run_command: CommandFn,
    sleeper: Callable[[float], None],
    clock: Callable[[], float],
    timeout: float,
) -> None:
    deadline = clock() + timeout
    while True:
        status = ace_status(session, base_url)
        used = _read_vram_mib(run_command)
        if not status["up"] and _memory_released(used):
            return
        if clock() >= deadline:
            raise GpuBusy("ACE-Step stopped but its VRAM was not released")
        sleeper(0.5)


def _wait_until_vram_free(
    run_command: CommandFn,
    sleeper: Callable[[float], None],
    clock: Callable[[], float],
    timeout: float,
) -> None:
    deadline = clock() + timeout
    while True:
        if _memory_released(_read_vram_mib(run_command)):
            return
        if clock() >= deadline:
            raise GpuBusy("GPU memory was not released")
        sleeper(0.5)


def release_for_video(
    log_fn: Optional[LogFn] = None,
    *,
    session: Any = None,
    zimage_loaded: Optional[Callable[[], bool]] = None,
    unload_zimage: Optional[Callable[[], None]] = None,
    load_zimage: Optional[Callable[[], None]] = None,
    run_command: CommandFn = _run_command,
    wait_timeout: float = 3600.0,
    vram_wait: float = VRAM_WAIT_SECONDS,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> HandoffState:
    """Stop ACE-Step and unload Z-Image. Raise ``GpuBusy`` if the card stays taken."""
    global _cooldown_until
    log = log_fn or _quiet
    if clock() < _cooldown_until:
        raise GpuBusy("GPU handoff failed recently; not retrying yet")
    if not _gpu_lock.acquire(timeout=wait_timeout):
        raise GpuBusy("the GPU is still in use by another job")
    state = HandoffState(holds_lock=True)
    try:
        if zimage_loaded is None or unload_zimage is None:
            module = _zimage_module()
            zimage_loaded = zimage_loaded or (lambda: module.RUNTIME.pipe is not None)
            unload_zimage = unload_zimage or module.unload
        state.zimage_was_loaded = bool(zimage_loaded())
        if state.zimage_was_loaded:
            log("Unloading resident Z-Image before the video job")
            unload_zimage()
        if session is None:
            import requests

            session = requests.Session()
        base = _ace_base()
        status = ace_status(session, base)
        state.ace_up = bool(status["up"])
        state.ace_model = str(status["loaded_model"] or "")
        if not state.ace_up:
            log("ACE-Step is not running")
            _cooldown_until = 0.0
            return state
        if state.ace_model and ace_http_unload(session, base):
            again = ace_status(session, base)
            state.ace_model = str(again["loaded_model"] or "")
            if not state.ace_model and _memory_released(_read_vram_mib(run_command)):
                state.ace_unloaded = True
                log("ACE-Step released its GPU weights")
                _cooldown_until = 0.0
                return state
        elif not state.ace_model and _memory_released(_read_vram_mib(run_command)):
            log("ACE-Step is up with no model loaded")
            _cooldown_until = 0.0
            return state
        stop_cmd = _stop_command()
        log("Stopping ACE-Step so Wan or LTX can use the GPU")
        if _returncode(run_command(stop_cmd)) != 0:
            raise GpuBusy(
                "ACE-Step is still holding the GPU. "
                f"The stop command {stop_cmd!r} failed."
            )
        state.ace_stopped = True
        state.ace_model = ""
        _wait_until_clear(session, base, run_command, sleeper, clock, vram_wait)
        log("ACE-Step released the GPU")
        _cooldown_until = 0.0
        return state
    except Exception as exc:
        if isinstance(exc, GpuBusy):
            _cooldown_until = clock() + HANDOFF_COOLDOWN_SECONDS
        restore_after_video(
            state,
            log_fn=log,
            session=session,
            load_zimage=load_zimage,
            run_command=run_command,
        )
        raise


def restore_after_video(
    state: HandoffState,
    log_fn: Optional[LogFn] = None,
    *,
    session: Any = None,
    load_zimage: Optional[Callable[[], None]] = None,
    run_command: CommandFn = _run_command,
) -> None:
    """Start ACE-Step again when this job stopped it. Runs on success and failure."""
    log = log_fn or _quiet
    try:
        if state.ace_stopped:
            # The 12 GB card cannot hold ACE-Step and Z-Image together. The
            # service comes back; the next image job loads Z-Image.
            log("Starting ACE-Step again")
            try:
                run_command(_start_command())
            except Exception as exc:
                log(f"ACE-Step start failed: {exc}")
            return
        if state.zimage_was_loaded:
            try:
                if load_zimage is None:
                    load_zimage = _zimage_module().load
                log("Restoring resident Z-Image")
                load_zimage()
            except Exception as exc:
                log(f"Z-Image restore failed: {exc}")
            return
        if state.ace_unloaded and state.ace_model and session is not None:
            load_path = os.environ.get("HAVNAI_ACESTEP_LOAD_PATH", "/v1/load_model").strip()
            try:
                session.post(
                    f"{_ace_base()}/{load_path.lstrip('/')}",
                    json={"model": state.ace_model},
                    timeout=60,
                )
                log(f"Asked ACE-Step to load {state.ace_model} again")
            except Exception as exc:
                log(f"ACE-Step reload failed: {exc}")
    finally:
        _release_lock(state)


def prepare_for_music(
    log_fn: Optional[LogFn] = None,
    *,
    session: Any = None,
    zimage_loaded: Optional[Callable[[], bool]] = None,
    unload_zimage: Optional[Callable[[], None]] = None,
    run_command: CommandFn = _run_command,
    wait_timeout: float = 3600.0,
    vram_wait: float = VRAM_WAIT_SECONDS,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> HandoffState:
    """Wait out a video job, then leave the card free for ACE-Step."""
    log = log_fn or _quiet
    if not _gpu_lock.acquire(timeout=wait_timeout):
        raise GpuBusy("a video job is still using the GPU")
    state = HandoffState(holds_lock=True)
    try:
        if zimage_loaded is None or unload_zimage is None:
            module = _zimage_module()
            zimage_loaded = zimage_loaded or (lambda: module.RUNTIME.pipe is not None)
            unload_zimage = unload_zimage or module.unload
        if zimage_loaded():
            log("Unloading resident Z-Image before the music job")
            unload_zimage()
        if session is None:
            import requests

            session = requests.Session()
        status = ace_status(session, _ace_base())
        if status["up"]:
            log("ACE-Step is already running")
            state.ace_up = True
            return state
        _wait_until_vram_free(run_command, sleeper, clock, vram_wait)
        start_cmd = _start_command()
        log("Starting ACE-Step")
        if _returncode(run_command(start_cmd)) != 0:
            raise GpuBusy(f"ACE-Step did not start ({start_cmd})")
        state.ace_up = True
        return state
    except Exception:
        _release_lock(state)
        raise


def finish_music(state: Optional[HandoffState]) -> None:
    """Release the card after a music job. ACE-Step keeps running."""
    if state is not None:
        _release_lock(state)
