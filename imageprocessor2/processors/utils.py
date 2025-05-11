from PIL import Image, ImageDraw
import os

import re

def classify_text(text):
    """
    Classify text as 'Math', 'Chemical', or 'Text'
    """
    math_patterns = [
        r'[\d\w\s\+\-\*\/\^\=\(\)\[\]]+',   # simple math patterns
        r'(sin|cos|tan|log|sqrt)',           # common math functions
    ]
    chemical_patterns = [
        r'^[A-Z][a-z]?[0-9]?[A-Z]?[a-z]?[0-9]*',  # basic chemical formulas
        r'->',                                    # chemical reaction arrow
    ]

    for pattern in chemical_patterns:
        if re.search(pattern, text):
            return 'Chemical'

    for pattern in math_patterns:
        if re.search(pattern, text):
            return 'Math'

    return 'Text'

class ImageUtils:
    @staticmethod
    def load_image(image_path):
        return Image.open(image_path).convert('RGB')

    @staticmethod
    def save_markdown(output_path, segments_data):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Image Analysis Output\n\n")
            for idx, segment in enumerate(segments_data, 1):
                f.write(f"## Segment {idx}\n")
                f.write(f"**Box**: ({segment['box'][0]}, {segment['box'][1]}) to ({segment['box'][2]}, {segment['box'][3]})\n\n")
                f.write(f"**Type**: {segment['type']}\n\n")
                f.write(f"**Content**: {segment['content']}\n\n")
                f.write("---\n")

    @staticmethod
    def save_debug_image(image_path, segments, output_path):
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        for box in segments:
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        image.save(output_path)