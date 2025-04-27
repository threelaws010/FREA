import csv
import cv2
import ast
from pathlib import Path

# Path to your unknowns CSV
csv_path = Path('E:/test/sample/unknown_segments.csv')

def review_unknowns(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_path = row['original_image']
            coords = ast.literal_eval(row['coordinates'])

            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Could not open {image_path}")
                continue

            # Draw rectangle around the unknown segment
            x1, y1, x2, y2 = coords
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, 'Unknown Segment', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Unknown Segment Review', img)
            print(f"Reviewing: {image_path} at {coords}")
            key = cv2.waitKey(0)

            if key == 27:  # ESC to quit early
                break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    review_unknowns(csv_path)
