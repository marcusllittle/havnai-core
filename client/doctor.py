"""Preflight diagnostics for a HavnAI node.

The node advertises the work it can take on. If it claims a capability it
cannot actually service, the coordinator hands it a job that fails at import
time and the operator sees a dead node with no explanation. This module exists
to catch that on the ground, before the node ever registers.

Checks are grouped by the capability they gate:

``core``
    Interpreter, dependencies and coordinator reachability. Nothing works
    without these.
``image``
    SDXL image generation - the baseline creator workload.
``face_swap``
    InstantID pipelines, the ``ip_adapter`` package and the antelopev2 face
    analysis pack.
``video``
    The ``engines`` package and whichever video runtime the node enables.

Run ``python -m client.doctor`` for a human report, or ``--json`` for a machine
readable one (the desktop app consumes the latter).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

OK = "ok"
WARN = "warn"
FAIL = "fail"

# Minimum free space for a creator node: one SDXL checkpoint plus working room.
MIN_FREE_GB = 25.0
MIN_PYTHON = (3, 10)

FEATURE_LABELS = {
    "core": "Core runtime",
    "image": "Image generation",
    "face_swap": "Face swap",
    "video": "Video generation",
}


@dataclass
class CheckResult:
    """A single diagnostic outcome."""

    id: str
    label: str
    status: str
    feature: str = "core"
    detail: str = ""
    remedy: str = ""

    @property
    def blocking(self) -> bool:
        return self.status == FAIL


@dataclass
class Report:
    """The full diagnostic run."""

    checks: List[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> CheckResult:
        self.checks.append(result)
        return result

    def by_feature(self, feature: str) -> List[CheckResult]:
        return [check for check in self.checks if check.feature == feature]

    def feature_ready(self, feature: str) -> bool:
        """A feature is ready when nothing gating it (or the core) failed."""

        if any(check.blocking for check in self.by_feature("core")):
            return False
        return not any(check.blocking for check in self.by_feature(feature))

    @property
    def capabilities(self) -> List[str]:
        return [
            feature
            for feature in ("image", "face_swap", "video")
            if self.feature_ready(feature)
        ]

    @property
    def healthy(self) -> bool:
        return not any(check.blocking for check in self.checks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "healthy": self.healthy,
            "capabilities": self.capabilities,
            "features": {
                feature: self.feature_ready(feature)
                for feature in ("image", "face_swap", "video")
            },
            "checks": [asdict(check) for check in self.checks],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_available(name: str) -> bool:
    """Whether a module can be imported without actually importing it."""

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _runtime_root() -> Path:
    """Directory that should contain the node runtime packages."""

    override = os.environ.get("HAVNAI_RUNTIME_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent.parent


def _havnai_home() -> Path:
    return Path(os.environ.get("HAVNAI_HOME") or (Path.home() / ".havnai")).expanduser()


def _server_url() -> str:
    return (
        os.environ.get("HAVNAI_SERVER_URL")
        or os.environ.get("SERVER_URL")
        or os.environ.get("COORDINATOR_URL")
        or "https://api.joinhavn.io"
    ).rstrip("/")


def _join_token() -> str:
    return (
        os.environ.get("JOIN_TOKEN")
        or os.environ.get("HAVNAI_NODE_TOKEN")
        or os.environ.get("SERVER_JOIN_TOKEN")
        or ""
    ).strip()


def _load_env_file() -> None:
    """Load ``~/.havnai/.env`` so the doctor sees what the node will see."""

    env_path = _havnai_home() / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Core checks
# ---------------------------------------------------------------------------


def check_python(report: Report) -> None:
    version = sys.version_info
    pretty = f"{version.major}.{version.minor}.{version.micro}"
    if version[:2] < MIN_PYTHON:
        report.add(
            CheckResult(
                "python_version",
                "Python interpreter",
                FAIL,
                detail=f"Python {pretty} is below the required {MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
                remedy=f"Install Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ and re-run the installer.",
            )
        )
        return
    report.add(CheckResult("python_version", "Python interpreter", OK, detail=f"Python {pretty}"))


def check_dependencies(report: Report) -> None:
    """Verify the Python packages each capability needs."""

    groups: List[tuple[str, str, List[str], str]] = [
        (
            "deps_core",
            "Core dependencies",
            ["requests", "numpy", "psutil", "PIL", "safetensors"],
            "core",
        ),
        (
            "deps_image",
            "Image dependencies",
            ["torch", "diffusers", "transformers", "accelerate"],
            "image",
        ),
        (
            "deps_face_swap",
            "Face swap dependencies",
            ["insightface", "cv2", "onnxruntime"],
            "face_swap",
        ),
    ]

    for check_id, label, modules, feature in groups:
        missing = [module for module in modules if not _module_available(module)]
        if missing:
            report.add(
                CheckResult(
                    check_id,
                    label,
                    FAIL,
                    feature=feature,
                    detail=f"missing: {', '.join(missing)}",
                    remedy=(
                        "Reinstall dependencies: "
                        "~/.havnai/venv/bin/pip install -r "
                        "~/.havnai/current/client/requirements-node.txt"
                    ),
                )
            )
        else:
            report.add(
                CheckResult(
                    check_id,
                    label,
                    OK,
                    feature=feature,
                    detail=f"{len(modules)} packages present",
                )
            )


def check_runtime_modules(report: Report) -> None:
    """Confirm the node has the runtime source files each capability imports.

    This is the check that catches a partial install: the client itself lands
    fine, then face swap and video die on import because their modules were
    never delivered.
    """

    root = _runtime_root()
    expectations: List[tuple[str, str, List[str], str]] = [
        (
            "runtime_face_swap",
            "Face swap modules",
            [
                "pipeline_stable_diffusion_xl_instantid.py",
                "pipeline_stable_diffusion_xl_instantid_inpaint.py",
                "ip_adapter/__init__.py",
                "ip_adapter/resampler.py",
                "ip_adapter/attention_processor.py",
                "ip_adapter/utils.py",
            ],
            "face_swap",
        ),
        (
            "runtime_video",
            "Video engine modules",
            [
                "engines/__init__.py",
                "engines/ltx_video/runner.py",
                "engines/ltx_video/config.py",
                "engines/ltx2/ltx2_runner.py",
                "engines/animatediff/animatediff_runner.py",
                "engines/wangp/runner.py",
            ],
            "video",
        ),
    ]

    # Face swap modules sit next to client.py; engines sit one level up.
    client_dir = root / "client" if (root / "client").is_dir() else root

    for check_id, label, relative_paths, feature in expectations:
        base = client_dir if feature == "face_swap" else root
        missing = [rel for rel in relative_paths if not (base / rel).exists()]
        if missing:
            report.add(
                CheckResult(
                    check_id,
                    label,
                    FAIL,
                    feature=feature,
                    detail=f"missing {len(missing)} file(s): {', '.join(missing[:3])}"
                    + ("…" if len(missing) > 3 else ""),
                    remedy=(
                        "Your node runtime is incomplete. Re-run the installer to fetch "
                        "the full bundle: curl -fsSL <server>/installers/install-node.sh | bash"
                    ),
                )
            )
        else:
            report.add(
                CheckResult(
                    check_id,
                    label,
                    OK,
                    feature=feature,
                    detail=f"{len(relative_paths)} modules present",
                )
            )


def check_gpu(report: Report) -> None:
    """Detect GPU availability and report usable VRAM."""

    if not shutil.which("nvidia-smi"):
        report.add(
            CheckResult(
                "gpu_driver",
                "GPU driver",
                WARN,
                detail="nvidia-smi not found - node will run CPU-only",
                remedy="Install the NVIDIA driver and CUDA runtime to serve creator jobs.",
            )
        )
    else:
        try:
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            detail = output.stdout.strip().splitlines()[0] if output.stdout.strip() else "detected"
            report.add(CheckResult("gpu_driver", "GPU driver", OK, detail=detail))
        except (subprocess.SubprocessError, OSError, IndexError):
            report.add(CheckResult("gpu_driver", "GPU driver", WARN, detail="nvidia-smi unreadable"))

    if not _module_available("torch"):
        # check_dependencies already reported this as a failure.
        return

    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            report.add(
                CheckResult(
                    "gpu_torch",
                    "CUDA available to PyTorch",
                    WARN,
                    feature="image",
                    detail="torch.cuda.is_available() is False",
                    remedy=(
                        "Install a CUDA build of PyTorch, e.g. "
                        "pip install torch --index-url https://download.pytorch.org/whl/cu124"
                    ),
                )
            )
            return
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        name = torch.cuda.get_device_name(0)
        status = OK if total_gb >= 11.0 else WARN
        report.add(
            CheckResult(
                "gpu_torch",
                "CUDA available to PyTorch",
                status,
                feature="image",
                detail=f"{name} ({total_gb:.1f} GB VRAM)",
                remedy="" if status == OK else "12 GB+ VRAM is recommended for video and face swap.",
            )
        )
    except Exception as exc:  # pragma: no cover - hardware dependent
        report.add(
            CheckResult(
                "gpu_torch",
                "CUDA available to PyTorch",
                WARN,
                feature="image",
                detail=f"probe failed: {exc}",
            )
        )


def check_disk(report: Report) -> None:
    home = _havnai_home()
    home.mkdir(parents=True, exist_ok=True)
    try:
        usage = shutil.disk_usage(home)
    except OSError as exc:
        report.add(CheckResult("disk_space", "Free disk space", WARN, detail=str(exc)))
        return

    free_gb = usage.free / (1024**3)
    status = OK if free_gb >= MIN_FREE_GB else FAIL
    report.add(
        CheckResult(
            "disk_space",
            "Free disk space",
            status,
            detail=f"{free_gb:.1f} GB free at {home}",
            remedy=""
            if status == OK
            else f"Free up space - a creator node needs at least {MIN_FREE_GB:.0f} GB for weights.",
        )
    )


def check_coordinator(report: Report) -> None:
    """Confirm the node can reach the coordinator and that its token works."""

    server = _server_url()
    if not _module_available("requests"):
        return

    import requests  # type: ignore

    try:
        response = requests.get(f"{server}/models/list", timeout=20)
    except requests.RequestException as exc:
        report.add(
            CheckResult(
                "coordinator_reachable",
                "Coordinator reachable",
                FAIL,
                detail=f"{server} unreachable: {exc}",
                remedy="Check the machine's outbound network access and SERVER_URL in ~/.havnai/.env.",
            )
        )
        return

    if response.status_code != 200:
        report.add(
            CheckResult(
                "coordinator_reachable",
                "Coordinator reachable",
                FAIL,
                detail=f"{server}/models/list returned HTTP {response.status_code}",
                remedy="Confirm SERVER_URL points at the coordinator API host.",
            )
        )
        return

    try:
        models = response.json().get("models", [])
    except ValueError:
        models = []
    report.add(
        CheckResult(
            "coordinator_reachable",
            "Coordinator reachable",
            OK,
            detail=f"{server} - {len(models)} models in manifest",
        )
    )

    token = _join_token()
    if not token:
        report.add(
            CheckResult(
                "join_token",
                "Join token",
                WARN,
                detail="no JOIN_TOKEN configured",
                remedy="Set JOIN_TOKEN in ~/.havnai/.env if the coordinator requires one.",
            )
        )
    else:
        report.add(
            CheckResult("join_token", "Join token", OK, detail=f"configured ({len(token)} chars)")
        )


def check_models(report: Report) -> None:
    """Report how many manifest models are actually on disk."""

    try:
        from client.model_sources import models_dir, requirements_from_manifest
    except ImportError:
        try:
            from model_sources import models_dir, requirements_from_manifest  # type: ignore
        except ImportError:
            return

    home = _havnai_home()
    target = models_dir(home)
    if not _module_available("requests"):
        return

    import requests  # type: ignore

    try:
        response = requests.get(f"{_server_url()}/models/list", timeout=20)
        response.raise_for_status()
        manifest = response.json().get("models", [])
    except Exception:
        # Coordinator reachability is already reported; don't double-fail here.
        local = list(target.glob("*.safetensors")) if target.exists() else []
        report.add(
            CheckResult(
                "models_present",
                "Model weights",
                OK if local else WARN,
                feature="image",
                detail=f"{len(local)} local checkpoint(s) in {target}",
            )
        )
        return

    plan = requirements_from_manifest(manifest, home)
    if not plan:
        report.add(
            CheckResult(
                "models_present",
                "Model weights",
                WARN,
                feature="image",
                detail="coordinator manifest lists no models",
            )
        )
        return

    present = [item for item in plan if item.present]
    manual = [item for item in plan if not item.present and not item.downloadable]

    if present:
        status = OK
        detail = f"{len(present)}/{len(plan)} manifest models in {target}"
    else:
        status = FAIL
        detail = f"no manifest models found in {target}"

    remedy = ""
    if len(present) < len(plan):
        remedy = "Fetch the rest with: python -m client.fetch_models"
    if manual and not present:
        remedy = (
            "These models must be supplied by you: "
            + ", ".join(item.filename for item in manual[:3])
        )

    report.add(
        CheckResult(
            "models_present",
            "Model weights",
            status,
            feature="image",
            detail=detail,
            remedy=remedy,
        )
    )


def check_face_assets(report: Report) -> None:
    """InstantID adapter + antelopev2 pack, both required for face swap."""

    cache = _havnai_home() / "instantid"
    adapter_candidates = [cache / "ip-adapter.bin", cache / "InstantID" / "ip-adapter.bin"]
    antelope = cache / "models" / "antelopev2"

    if any(path.exists() for path in adapter_candidates):
        report.add(
            CheckResult(
                "face_adapter",
                "InstantID adapter",
                OK,
                feature="face_swap",
                detail=f"cached in {cache}",
            )
        )
    else:
        report.add(
            CheckResult(
                "face_adapter",
                "InstantID adapter",
                WARN,
                feature="face_swap",
                detail="not cached yet",
                remedy="Downloads automatically from Hugging Face on the first face swap job.",
            )
        )

    if antelope.exists() and any(antelope.glob("*.onnx")):
        report.add(
            CheckResult(
                "face_analysis",
                "antelopev2 face pack",
                OK,
                feature="face_swap",
                detail=str(antelope),
            )
        )
    else:
        report.add(
            CheckResult(
                "face_analysis",
                "antelopev2 face pack",
                WARN,
                feature="face_swap",
                detail="not present",
                remedy="Fetch it with: python -m client.fetch_models --face-assets",
            )
        )


def check_video_runtime(report: Report) -> None:
    """Report which video runtime, if any, this node can drive."""

    isolated = os.environ.get("HAVNAI_LTX_VIDEO_PYTHON", "").strip()
    if isolated:
        if Path(isolated).exists():
            report.add(
                CheckResult(
                    "video_runtime",
                    "Video runtime",
                    OK,
                    feature="video",
                    detail=f"isolated interpreter at {isolated}",
                )
            )
        else:
            report.add(
                CheckResult(
                    "video_runtime",
                    "Video runtime",
                    FAIL,
                    feature="video",
                    detail=f"HAVNAI_LTX_VIDEO_PYTHON points at a missing path: {isolated}",
                    remedy="Correct or unset HAVNAI_LTX_VIDEO_PYTHON in ~/.havnai/.env.",
                )
            )
        return

    report.add(
        CheckResult(
            "video_runtime",
            "Video runtime",
            OK if _module_available("torch") else WARN,
            feature="video",
            detail="in-process (shared venv)",
            remedy=""
            if _module_available("torch")
            else "Install torch to enable the in-process video runtime.",
        )
    )


def check_outputs(report: Report) -> None:
    outputs = Path(
        os.environ.get("HAVNAI_OUTPUTS_DIR") or (_havnai_home() / "outputs")
    ).expanduser()
    try:
        outputs.mkdir(parents=True, exist_ok=True)
        probe = outputs / ".havnai-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        report.add(CheckResult("outputs_writable", "Outputs directory", OK, detail=str(outputs)))
    except OSError as exc:
        report.add(
            CheckResult(
                "outputs_writable",
                "Outputs directory",
                FAIL,
                detail=f"{outputs} is not writable: {exc}",
                remedy="Fix directory permissions or set HAVNAI_OUTPUTS_DIR to a writable path.",
            )
        )


CHECKS: List[Callable[[Report], None]] = [
    check_python,
    check_dependencies,
    check_runtime_modules,
    check_gpu,
    check_disk,
    check_outputs,
    check_coordinator,
    check_models,
    check_face_assets,
    check_video_runtime,
]


def run(*, skip_network: bool = False) -> Report:
    """Execute every diagnostic and return the collected report."""

    _load_env_file()
    report = Report()
    network_checks = {check_coordinator, check_models}
    for check in CHECKS:
        if skip_network and check in network_checks:
            continue
        try:
            check(report)
        except Exception as exc:  # a broken check must not hide the others
            report.add(
                CheckResult(
                    getattr(check, "__name__", "check"),
                    getattr(check, "__name__", "check"),
                    WARN,
                    detail=f"diagnostic raised {type(exc).__name__}: {exc}",
                )
            )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_ICONS = {OK: "  OK  ", WARN: " WARN ", FAIL: " FAIL "}


def _supports_colour() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def format_report(report: Report) -> str:
    colours = {OK: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m"}
    reset = "\033[0m"
    use_colour = _supports_colour()

    lines: List[str] = ["", "HavnAI node preflight", "=" * 58]
    for feature in ("core", "image", "face_swap", "video"):
        checks = report.by_feature(feature)
        if not checks:
            continue
        lines.append("")
        header = FEATURE_LABELS[feature]
        if feature != "core":
            header += " — " + ("READY" if report.feature_ready(feature) else "NOT READY")
        lines.append(header)
        lines.append("-" * 58)
        for check in checks:
            icon = _ICONS[check.status]
            if use_colour:
                icon = f"{colours[check.status]}{icon}{reset}"
            lines.append(f"[{icon}] {check.label}: {check.detail}")
            if check.remedy and check.status != OK:
                lines.append(f"         → {check.remedy}")

    lines.append("")
    lines.append("=" * 58)
    capabilities = report.capabilities
    lines.append(
        "This node can serve: " + (", ".join(capabilities) if capabilities else "nothing yet")
    )
    if not report.healthy:
        blocking = [check.label for check in report.checks if check.blocking]
        lines.append(f"Blocking issues: {', '.join(blocking)}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="havnai-doctor",
        description="Diagnose a HavnAI node install and report what it can serve.",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--offline", action="store_true", help="skip checks that contact the coordinator"
    )
    args = parser.parse_args(argv)

    report = run(skip_network=args.offline)

    if args.json:
        payload = report.to_dict()
        payload["host"] = socket.gethostname()
        print(json.dumps(payload, indent=2))
    else:
        print(format_report(report))

    return 0 if report.healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
