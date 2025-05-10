
from PIL import Image
import os

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
                f.write(f"**Type**: {segment['type']}\n\n")
                f.write(f"**Content**: {segment['content']}\n\n")
                f.write("---\n")