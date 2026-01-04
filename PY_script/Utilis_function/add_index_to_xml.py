import sys
import os
import xml.etree.ElementTree as ET
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from configs.config import Config

# Root folder of your dataset
xml_root_folder = Config.dataset_root
write_to_csv_path = Config.csv_index_path

# Dictionary for brand -> index mapping
brand_to_index = {}
current_index = 0

def process_xml(xml_path):
    global current_index
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    for obj in root.findall('object'):
        brand_name = obj.find('name').text
        
        # Assign index if not seen before
        if brand_name not in brand_to_index:
            brand_to_index[brand_name] = current_index
            current_index += 1
        
        # Add <index> tag or update existing
        index_tag = obj.find('index')
        if index_tag is None:
            index_tag = ET.SubElement(obj, 'index')
        index_tag.text = str(brand_to_index[brand_name])
    
    # Save XML back
    tree.write(xml_path)
    print(f"Processed {xml_path}")

# # Walk through the dataset
# for category in os.listdir(xml_root_folder):
#     category_path = os.path.join(xml_root_folder, category)
#     if not os.path.isdir(category_path):
#         continue
#     for brand in os.listdir(category_path):
#         brand_path = os.path.join(category_path, brand)
#         if not os.path.isdir(brand_path):
#             continue
#         for xml_file in os.listdir(brand_path):
#             if xml_file.endswith('.xml'):
#                 xml_path = os.path.join(brand_path, xml_file)
#                 process_xml(xml_path)

# Save brand -> index mapping to CSV
csv_path = os.path.join(write_to_csv_path, "brand_to_index.csv")
with open(csv_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["brand", "index"])
    for brand, idx in sorted(brand_to_index.items(), key=lambda x: x[1]):
        writer.writerow([brand, idx])

print(f"Brand-to-index mapping saved to {csv_path}")
