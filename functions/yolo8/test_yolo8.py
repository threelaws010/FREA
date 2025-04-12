from segmenter import segment_image
import shutil
#shutil.rmtree(result["temp_dir"])

result = segment_image(r"E:/Astronomy/Envelope102/Image (16).jpg")

print("Segmented image saved to:", result["segmented_image_path"])
print("Temporary directory:", result["temp_dir"])
print("Detected objects:")
for box in result["boxes"]:
    class_name = result["class_names"][box["class_id"]]
    print(f" - {class_name} ({box['confidence']:.2f}): {box}")
