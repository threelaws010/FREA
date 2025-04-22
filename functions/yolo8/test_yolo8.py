from segmenter import segment_image
import shutil
import cv2
#shutil.rmtree(result["temp_dir"])

result = segment_image(r"E:/Astronomy/Envelope102/Image (16).jpg")

print("Segmented image saved to:", result["segmented_image_path"])
print("Temporary directory:", result["temp_dir"])
print("Detected objects:")
for box in result["boxes"]:
    class_name = result["class_names"][box["class_id"]]
    print(f" - {class_name} ({box['confidence']:.2f}): {box}")


def test_segment_image_integration():
    # Create a dummy image for segmentation
    test_img = np.full((256, 256, 3), 255, dtype=np.uint8)
    img_path = Path(tempfile.gettempdir()) / "integration_test_img.jpg"
    cv2.imwrite(str(img_path), test_img)

    # Run real segmentation (requires model in cwd or known path)
    result = segment_image(str(img_path))

    assert "boxes" in result
    assert "masks" in result
    assert isinstance(result["boxes"], list)

