
import os, json, requests
from pathlib import Path
from typing import Dict, Any, Optional

def _get(url: str, key: str):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()

def _extract_embed_model(obj: Dict[str, Any]) -> Optional[str]:
    # defensively check common shapes/keys across AnythingLLM versions
    return (
        obj.get("embeddingModel")
        or (obj.get("config") or {}).get("embeddingModel")
        or (obj.get("settings") or {}).get("embeddingModel")
        or obj.get("embedModel")
        or (obj.get("config") or {}).get("embedModel")
        or (obj.get("settings") or {}).get("embedModel")
    )

def read_embed_model_from_api(base: str, key: str, slug: str) -> Optional[str]:
    """
    Try to read the embedding model from AnythingLLM's API for a workspace,
    falling back to system settings if needed.
    """
    base = base.rstrip("/")
    # Try common workspace endpoints
    candidates = (
        f"/api/v1/workspace/{slug}",
        f"/api/v1/workspace/{slug}/settings",
        f"/api/v1/workspace/{slug}/config",
    )
    for path in candidates:
        try:
            data = _get(base + path, key)
            val = _extract_embed_model(data)
            if val:
                return val
        except Exception:
            pass

    # Fallback: system-level settings
    try:
        data = _get(base + "/api/v1/system/settings", key)
        val = _extract_embed_model(data)
        if val:
            return val
    except Exception:
        pass

    return None

def read_embed_model_from_desktop(store: str = "~/.config/anythingllm-desktop/storage") -> Optional[str]:
    """
    Desktop fallback (Linux): read local storage files to find the embedding model.
    Looks under storage/workspaces/*/(settings|config|workspace|meta).json
    """
    wdir = Path(store).expanduser() / "workspaces"
    if not wdir.exists():
        return None
    for d in wdir.iterdir():
        if not d.is_dir():
            continue
        for name in ("settings.json", "config.json", "workspace.json", "meta.json"):
            p = d / name
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    val = _extract_embed_model(data)
                    if val:
                        return val
                except Exception:
                    pass
    return None

if __name__ == "__main__":
    BASE = os.getenv("ANYLLM_BASE", "http://localhost:3001")
    KEY  = os.getenv("ANYLLM_API_KEY", "")
    SLUG = os.getenv("ANYLLM_WORKSPACE", "default")

    model = read_embed_model_from_api(BASE, KEY, SLUG) or read_embed_model_from_desktop()
    print(model or "")
