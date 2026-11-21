"""Free the 12 GB card before a Wan or LTX-2.3 job, then put residents back.

Z-Image stays loaded between image jobs and is about 10 GB. ACE-Step is a
separate process on port 8001 and also keeps weights on the card. A Wan 2.2
clip peaks at 10.7 to 11.2 GB, so neither can stay resident. LTX-2.3 peaks
near 4.7 GB, which still does not fit beside Z-Image.

After the clip, Z-Image is loaded again when it was the resident model.
ACE-Step is started again when this job stopped its process. The two weight
sets are not both put back on the card: if Z-Image returns, ACE-Step is left
running only when its start command does not have to be issued, or the
process is started and the weight load is left for the next music job.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Optional


class GpuBusy(RuntimeError):
    """A resident model is still on the card, so the video job must not start."""


LogFn = Callable[[str], None]

_UNLOAD_PATHS = ("/v1/unload_model", "/v1/models/unload", "/unload_model")


@dataclass
class HandoffState:
    zimage_was_loaded: bool = False
    ace_up: bool = False
    ace_model: str = ""
    ace_stopped: bool = False
    ace_unloaded: bool = False


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


def _run_command(command: str) -> int:
    completed = subprocess.run(command, shell=True, check=False)
    return int(completed.returncode)


def release_for_video(
    log_fn: Optional[LogFn] = None,
    *,
    session: Any = None,
    zimage_loaded: Optional[Callable[[], bool]] = None,
    unload_zimage: Optional[Callable[[], None]] = None,
    load_zimage: Optional[Callable[[], None]] = None,
    run_command: Callable[[str], int] = _run_command,
) -> HandoffState:
    """Unload Z-Image and pause ACE-Step. Raise ``GpuBusy`` if ACE stays loaded."""
    log = log_fn or _quiet
    state = HandoffState()
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
            return state
        if not state.ace_model:
            log("ACE-Step is up with no model loaded")
            return state
        log(f"Unloading ACE-Step model {state.ace_model}")
        if ace_http_unload(session, base):
            again = ace_status(session, base)
            if not again["loaded_model"]:
                state.ace_unloaded = True
                log("ACE-Step released its GPU weights")
                return state
        stop_cmd = os.environ.get("HAVNAI_ACESTEP_STOP_CMD", "").strip()
        if stop_cmd:
            log("Stopping ACE-Step so Wan or LTX can use the GPU")
            if run_command(stop_cmd) == 0:
                state.ace_stopped = True
                state.ace_model = ""
                return state
        raise GpuBusy(
            "ACE-Step is still holding the GPU. Stop it, or set "
            "HAVNAI_ACESTEP_STOP_CMD to a command that stops it, before a "
            "Wan 2.2 or LTX-2.3 job."
        )
    except Exception:
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
    run_command: Callable[[str], int] = _run_command,
) -> None:
    """Put back the resident that was actually loaded before the clip."""
    log = log_fn or _quiet
    start_cmd = os.environ.get("HAVNAI_ACESTEP_START_CMD", "").strip()
    # Z-Image is the model the image worker expects to find warm. Start
    # ACE-Step's process only when Z-Image is not coming back, so the 12 GB
    # card is not asked to hold both.
    if state.zimage_was_loaded:
        if state.ace_stopped and start_cmd:
            log("Leaving ACE-Step stopped because Z-Image is being restored")
        try:
            if load_zimage is None:
                load_zimage = _zimage_module().load
            log("Restoring resident Z-Image")
            load_zimage()
        except Exception as exc:
            log(f"Z-Image restore failed: {exc}")
        return
    if state.ace_stopped and start_cmd:
        log("Starting ACE-Step again")
        try:
            run_command(start_cmd)
        except Exception as exc:
            log(f"ACE-Step start failed: {exc}")
    elif state.ace_unloaded and state.ace_model and session is not None:
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
