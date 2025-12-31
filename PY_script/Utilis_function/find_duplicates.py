import os
import glob
import xml.etree.ElementTree as ET
import csv

def find_images_with_multiple_bboxes(root_dir, output_file="images_with_multiple_bboxes.csv"):
    """
    Traverse the dataset directory and find images that have more than one bounding box.
    Prints image path and associated labels, and writes to a CSV file.
    """
    results = []

    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category)
        if not os.path.isdir(category_path):
            continue

        for brand in os.listdir(category_path):
            brand_path = os.path.join(category_path, brand)
            if not os.path.isdir(brand_path):
                continue

            # Iterate over all images in this brand folder
            for img_path in glob.glob(os.path.join(brand_path, "*.jpg")):
                xml_path = img_path.replace(".jpg", ".xml")
                if not os.path.exists(xml_path):
                    print(f"Warning: XML not found for {img_path}")
                    continue

                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    objects = root.findall("object")

                    if len(objects) > 1:
                        labels = [obj.find("name").text for obj in objects]
                        print(f"Image: {img_path}")
                        print(f"Labels: {labels}\n")
                        results.append([img_path, ";".join(labels)])

                except Exception as e:
                    print(f"Error parsing XML {xml_path}: {e}")

    # Write results to CSV
    if results:
        with open(output_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "labels"])  # header
            writer.writerows(results)
        print(f"Results saved to {output_file}")
    else:
        print("No images with multiple bounding boxes found.")

if __name__ == "__main__":
    dataset_dir = "LogoDet-3K/LogoDet-3K"  # replace with your dataset root
    find_images_with_multiple_bboxes(dataset_dir)
