
import os, json, requests
from pathlib import Path

BASE = os.getenv("ANYLLM_BASE", "http://localhost:3001")
KEY  = os.getenv("ANYLLM_API_KEY", "brx-JJDATRX-WPF4TB2-J3384JA-PXBZ4GP")
HEAD = {"Authorization": f"Bearer {KEY}"} if KEY else {}

def api_list():
    try:
        r = requests.get(f"{BASE.rstrip('/')}/api/v1/workspaces", headers=HEAD, timeout=10)
        r.raise_for_status()
        data = r.json()
        items = data.get("workspaces") or data.get("items") or data
        if not isinstance(items, list):
            return []
        out = []
        for w in items:
            slug = (w.get("slug") or w.get("workspaceSlug") or w.get("id") or "").strip()
            name = (w.get("name") or w.get("workspaceName") or "").strip()
            out.append({"slug": slug, "name": name})
        return out
    except Exception:
        return []

def desktop_list(store="~/.config/anythingllm-desktop/storage"):
    base = Path(store).expanduser() / "workspaces"
    if not base.exists():
        return []
    out = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        for fname in ("workspace.json","settings.json","config.json","meta.json"):
            p = d / fname
            if not p.exists():
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                slug = (obj.get("slug") or obj.get("workspaceSlug") or obj.get("id") or "").strip()
                name = (obj.get("name") or obj.get("workspaceName") or "").strip()
                if slug or name:
                    out.append({"slug": slug, "name": name, "source": str(p)})
                    break
            except Exception:
                pass
    return out

def main():
    rows = api_list()
    if not rows:
        rows = desktop_list()
    if not rows:
        print("No workspaces found via API or Desktop storage.")
        return
    print("Available Workspaces:")
    for i, r in enumerate(rows, 1):
        src = r.get("source","API")
        print(f"{i:2d}. slug='{r.get('slug','')}'  name='{r.get('name','')}'  ({src})")

if __name__ == "__main__":
    main()
