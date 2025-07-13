import os
import requests
from bs4 import BeautifulSoup
import markdownify
from zipfile import ZipFile

# === CONFIG ===
URL = "https://enzmannarchive.org/about-frea/"
OUTPUT_DIR = "frea_markdown"
MARKDOWN_FILENAME = "about-frea.md"
ZIP_FILENAME = "frea_markdown.zip"

# === CREATE OUTPUT DIR ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === FETCH HTML ===
print(f"Fetching {URL}...")
response = requests.get(URL)
response.raise_for_status()

# === PARSE HTML ===
soup = BeautifulSoup(response.text, "html.parser")

# Remove unwanted elements like nav, footer, scripts
for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
    tag.decompose()

# Optional: focus only on <main> content if it exists
main_content = soup.find("main") or soup.body

# === CONVERT TO MARKDOWN ===
markdown_text = markdownify.markdownify(str(main_content), heading_style="ATX")

# Add YAML frontmatter
frontmatter = f"""---
title: "About FREA"
source_url: "{URL}"
---

"""
markdown_text = frontmatter + markdown_text

# === WRITE MARKDOWN FILE ===
md_path = os.path.join(OUTPUT_DIR, MARKDOWN_FILENAME)
with open(md_path, "w", encoding="utf-8") as f:
    f.write(markdown_text)

print(f"Saved Markdown to {md_path}")

# === ZIP THE RESULT ===
with ZipFile(ZIP_FILENAME, "w") as zipf:
    zipf.write(md_path, arcname=MARKDOWN_FILENAME)

print(f"Zipped to {ZIP_FILENAME}")
