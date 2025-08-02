import os
import openai
from dotenv import load_dotenv
from PIL import Image
import base64
from openai import OpenAI


# Load environment variables
load_dotenv()
input_dir = os.getenv("INPUT_DIR")
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# Set model name
#MODEL = "gpt-4o"
MODEL = "gpt-4o-mini"  # Use the mini version for lower costs and faster processing
#MODEL ="gpt-4-turbo"
TOKENS = 2048  # Adjust based on your model's token limit
BATCH_LIMIT = 100

# Prompt to guide the model
VISION_PROMPT = (
    "You are analyzing a scanned image. Extract all typed and handwritten text. "
    "Describe any mathematical formulas, chemical structures, and illustrations. "
    "Also explain the visual layout if relevant."
)

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def process_image(image_path):
    image_base64 = encode_image_to_base64(image_path)
    image_filename = os.path.basename(image_path)  # Extract the image name

    # Inject the image filename into the system prompt
    system_prompt = (
        f"You are given a JPEG file named \"{image_filename}\" containing a complex document. "
        "This document may include:\n"
        "- Typed text (including structured text such as tables, bullet points, headings, etc.)\n"
        "- Handwritten annotations or notes\n"
        "- Images or illustrations (e.g., diagrams, charts)\n"
        "- Mathematical formulas (typed or handwritten)\n"
        "- Chemical formulas or molecular diagrams\n\n"
        "Please perform the following tasks:\n\n"
        "1. **OCR and Transcription**:\n"
        "   - Accurately extract and transcribe all visible text, both printed and handwritten.\n"
        "   - Maintain the original reading order and logical structure where possible (e.g., sections, headers, subheaders).\n"
        "   - Include any visible mathematical and chemical formulas using appropriate notation (LaTeX-style preferred if available).\n\n"
        "2. **Image and Illustration Descriptions**:\n"
        "   - Detect and describe any non-textual visual content such as illustrations, diagrams, graphs, photos, or sketches.\n"
        "   - For each image, provide a short textual description (e.g., “Illustration of a molecule with six carbon atoms,” or “Bar chart showing population trends”).\n\n"
        "3. **Unusual Content and Anomalies**:\n"
        "   - Make special note of any unusual or ambiguous content such as:\n"
        "     - Illegible handwriting\n"
        "     - Crossed-out sections\n"
        "     - Symbols or characters that are unclear or rare\n"
        "     - Content that seems out of place or inconsistent with the rest of the document\n"
        "   - If possible, include approximate coordinates for each anomaly in the form: "
        "{x: ..., y: ..., width: ..., height: ...}.\n\n"
        "4. **Summary**:\n"
        "   - Provide a concise summary of the document purpose, main topics covered, and any apparent conclusions or key points.\n\n"
        "5. **Image Reference**:\n"
        f"   - At the top of the response, include the original image filename: {image_filename}\n\n"
        "Output Format:\n"
        "- Sectioned output with clear headings for each part:\n"
        "  - Image Reference\n"
        "  - Transcribed Text\n"
        "  - Image Descriptions\n"
        "  - Unusual Content\n"
        "  - Document Summary"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        max_tokens=TOKENS,
        temperature=0.2
    )

    return response.choices[0].message.content



def main():
    if not input_dir or not os.path.isdir(input_dir):
        print("INPUT_DIR is not set or invalid.")
        return

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".jpg")]
    if BATCH_LIMIT:
        image_files = image_files[:BATCH_LIMIT]

    for file_name in image_files:
        image_path = os.path.join(input_dir, file_name)
        print(f"Processing: {file_name}")
        try:
            output_text = process_image(image_path)
            rel_path = os.path.relpath(image_path, input_dir)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            folder_parts = os.path.abspath(os.path.dirname(image_path)).split(os.sep)
            folder_name_part = "_".join(folder_parts[-3:])  # Adjust depth as needed
            output_file_name = f"{MODEL}_{TOKENS}_{folder_name_part}_{base_name}.txt"
            output_path = os.path.join(os.path.dirname(image_path), output_file_name)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output_text)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Failed to process {file_name}: {e}")


if __name__ == "__main__":
    main()
