
import os
import base64
from pathlib import Path
from dotenv import load_dotenv

# OpenAI-compatible SDK (works with LM Studio when base_url is set)
from openai import OpenAI

# Optional: Pillow just to verify image files exist/can open (not required to run)
try:
    from PIL import Image  # noqa: F401
except Exception:
    pass

"""
gpt_process_images_lmstudio.py
--------------------------------
Processes JPG images in INPUT_DIR (recursively) using a *local* model served by
LM Studio's OpenAI-compatible API. Produces structured text outputs similar to
the original script, but without calling OpenAI's hosted API.

Key behaviors (preserved):
  - Recursively walks INPUT_DIR to find *.jpg files in all subfolders.
  - Writes outputs next to each input image.
  - Honors BATCH_LIMIT and OVERRIDE_EXISTING.

Additional behavior (requested):
  - The output text filename includes the image's relative folder path components.

Requirements:
  - LM Studio Developer Server running (http://localhost:1234/v1 by default)
  - A vision-capable model loaded in LM Studio (e.g., LLaVA, Llama 3.2 Vision)

.env keys respected:
  INPUT_DIR               Directory containing .jpg images (searches recursively)
  OPENAI_BASE_URL         e.g. http://localhost:1234/v1  (LM Studio default)
  OPENAI_API_KEY          e.g. lm-studio                  (LM Studio default)
  MODEL                   The exact model name as shown in LM Studio (must support images)
  TOKENS                  Max tokens (default 2048)
  BATCH_LIMIT             Max number of images processed per run (default 3)
  OVERRIDE_EXISTING       "true"/"false" to overwrite existing outputs (default false)
"""

load_dotenv()

INPUT_DIR = os.getenv("INPUT_DIR", "").strip()
BASE_URL  = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1").strip()
API_KEY   = os.getenv("OPENAI_API_KEY", "lm-studio").strip()

# Example models (must be available/loaded in LM Studio):
#   - "meta-llama/Llama-3.2-11B-Vision-Instruct"
#   - "llava:7b"   (naming depends on how your LM Studio lists it)
MODEL     = os.getenv("MODEL", "meta-llama/Llama-3.2-11B-Vision-Instruct").strip()

TOKENS        = int(os.getenv("TOKENS", "2048"))
BATCH_LIMIT   = int(os.getenv("BATCH_LIMIT", "3"))
OVERRIDE      = os.getenv("OVERRIDE_EXISTING", "false").lower() in ("1", "true", "yes")

VISION_PROMPT = (
    "You are analyzing a scanned image. Extract all typed and handwritten text. "
    "Describe any mathematical formulas, chemical structures, and illustrations. "
    "Also explain the visual layout if relevant."
)

SYSTEM_TEMPLATE = """You are given a JPEG file named "{filename}" containing a complex document.
This document may include:
- Typed text (including structured text such as tables, bullet points, headings, etc.)
- Handwritten annotations or notes
- Images or illustrations (e.g., diagrams, charts)
- Mathematical formulas (typed or handwritten)
- Chemical formulas or molecular diagrams

Please perform the following tasks:

1. **OCR and Transcription**:
   - Accurately extract and transcribe all visible text, both printed and handwritten.
   - Maintain the original reading order and logical structure where possible (e.g., sections, headers, subheaders).
   - Include any visible mathematical and chemical formulas using appropriate notation (LaTeX-style preferred if available).

2. **Image and Illustration Descriptions**:
   - Detect and describe any non-textual visual content such as illustrations, diagrams, graphs, photos, or sketches.
   - For each image, provide a short textual description.

3. **Unusual Content and Anomalies**:
   - Note any illegible handwriting, crossed-out sections, rare symbols, or inconsistencies.
   - If possible, include approximate coordinates: {{x: ..., y: ..., width: ..., height: ...}}.

4. **Summary**:
   - Provide a concise summary of the document purpose, main topics, and key points.

5. **Image Reference**:
   - At the top, include the original image filename: {filename}

Output:
- Use clear section headings:
  - Image Reference
  - Transcribed Text
  - Image Descriptions
  - Unusual Content
  - Document Summary
"""

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def ensure_dir(path: str | Path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _sanitize_component(s: str) -> str:
    """
    Make a path component safe for filenames:
      - keep alphanumerics, dash, underscore, and dot
      - replace everything else with a single dash
      - collapse repeated dashes
    """
    import re
    # Replace non-safe chars with '-'
    s = re.sub(r'[^A-Za-z0-9._-]+', '-', s)
    # Collapse multiple dashes
    s = re.sub(r'-{2,}', '-', s).strip('-')
    # Avoid empty component
    return s or "x"

def find_jpgs_recursive(root_dir: str):
    """
    Yield absolute file paths for *.jpg files under root_dir (recursive).
    Note: Only 'jpg' extension is searched. Lowercase comparison.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".jpg"):
                yield os.path.join(dirpath, fname)

def main():
    if not INPUT_DIR:
        print("❌ INPUT_DIR is not set. Put it in your .env or export it.")
        return
    if not os.path.isdir(INPUT_DIR):
        print(f"❌ INPUT_DIR does not exist: {INPUT_DIR}")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    images = list(find_jpgs_recursive(INPUT_DIR))
    if not images:
        print(f"ℹ️ No .jpg files found under {INPUT_DIR} (searched recursively)")
        return

    print(f"🔎 Found {len(images)} .jpg file(s) under {INPUT_DIR}")

    processed = 0
    for image_path in images:
        file_name = os.path.basename(image_path)
        base = Path(file_name).stem

        # Compute a safe representation of the relative folder path
        rel_dir = os.path.relpath(os.path.dirname(image_path), INPUT_DIR)
        if rel_dir in (".", ""):
            rel_tag = ""
        else:
            parts = [p for p in rel_dir.split(os.sep) if p not in (".", "")]
            safe_parts = [_sanitize_component(p) for p in parts]
            rel_tag = "__".join(safe_parts)

        # Write outputs alongside the input file, including folder path in the name
        model_tag = Path(MODEL).name.replace("/", "_")
        prefix = f"{model_tag}_{TOKENS}_" + (f"{rel_tag}__" if rel_tag else "")
        output_name = f"{prefix}{base}.txt"
        output_path = os.path.join(os.path.dirname(image_path), output_name)

        if os.path.exists(output_path) and not OVERRIDE:
            print(f"↩️  Skipping (exists): {image_path}")
            continue

        print(f"🖼️  Processing: {image_path} with model '{MODEL}' via {BASE_URL}")
        try:
            img_b64 = encode_image_to_base64(image_path)
            system_prompt = SYSTEM_TEMPLATE.format(filename=file_name)

            # Build OpenAI-compatible, multi-part user content
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    ],
                },
            ]

            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=TOKENS,
                temperature=0.2,
            )

            text = resp.choices[0].message.content if resp.choices else ""
            ensure_dir(Path(output_path).parent)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text or "")
            print(f"✅ Saved: {output_path}")
            processed += 1

        except Exception as e:
            print(f"❌ Failed on {image_path}: {e}")

        if BATCH_LIMIT and processed >= BATCH_LIMIT:
            print(f"⏹️  Reached BATCH_LIMIT={BATCH_LIMIT}")
            break

if __name__ == "__main__":
    main()
