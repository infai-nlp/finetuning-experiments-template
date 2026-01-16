from pathlib import Path
from typing import Any, Dict

import yaml


def load_cfg(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML experiment config file.
    Automatically expands user (~) and relative paths.
    Leaves everything as plain dicts (no dataclasses).
    """

    path = Path(path).expanduser()

    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    def fix(o, key=None):
        """Recursively expand filesystem paths."""
        if isinstance(o, dict):
            return {k: fix(v, k) for k, v in o.items()}
        if isinstance(o, list):
            return [fix(v, key) for v in o]
        if isinstance(o, str):
            # treat anything looking like a path (except model names) as a path
            if key not in {"model_name"} and ("/" in o or o.startswith((".", "~"))):
                return str(Path(o).expanduser())
        return o

    return fix(cfg)
