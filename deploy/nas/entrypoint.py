"""Start the prepared model directly, without setup downloads or automatic model changes."""
import json
import os
from pathlib import Path
import sys

if not os.environ.get("STRATA_API_KEY", "").strip():
    raise SystemExit("STRATA_API_KEY must be set for LAN serving")
config_path = Path("/config/strata.json")
cfg = json.loads(config_path.read_text())
required = [cfg["exe"], Path(cfg["tokenizer"]) / "vocab.json"]
for flag in ("--pack", "--native", "--ple-gguf", "--expert-profile", "--mtp"):
    index = cfg["args"].index(flag)
    required.append(cfg["args"][index + 1])
for path in required:
    if not Path(path).exists():
        raise SystemExit(f"Required model artifact is missing: {path}")
os.chdir("/opt/strata")
os.execv(sys.executable, [sys.executable, "-m", "serve.server", "--engine", "strata",
                        "--config", str(config_path), "--host", "0.0.0.0", "--port", "8080"])
