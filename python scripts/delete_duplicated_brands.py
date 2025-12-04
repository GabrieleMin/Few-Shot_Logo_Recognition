import os
import xml.etree.ElementTree as ET
import re
import shutil

def clean_duplicate_brand_folders(root_dir):
    """
    Keep only the first duplicate brand folder, rename it to the base brand name,
    and delete the rest. Fix XML labels in the kept folder to remove suffixes like -1, -2, etc.
    """
    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category)
        if not os.path.isdir(category_path):
            continue

        # Find brand folders with '-number' suffix
        brand_folders = [b for b in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, b))]
        base_name_to_folders = {}

        for b in brand_folders:
            match = re.match(r"^(.*?)-\d+$", b)
            if match:
                base_name = match.group(1)
                if base_name not in base_name_to_folders:
                    base_name_to_folders[base_name] = []
                base_name_to_folders[base_name].append(b)

        # Process duplicates
        for base_name, folders in base_name_to_folders.items():
            if len(folders) < 1:
                continue  # no duplicates

            folders.sort()  # keep the first folder alphabetically
            folder_to_keep = folders[0]
            folders_to_delete = folders[1:]

            kept_folder_path = os.path.join(category_path, folder_to_keep)
            new_folder_path = os.path.join(category_path, base_name)

            print(f"Keeping folder: {folder_to_keep}, renaming to: {base_name}, deleting duplicates: {folders_to_delete}")

            # Fix XML labels in the kept folder
            for xml_file in os.listdir(kept_folder_path):
                if xml_file.endswith(".xml"):
                    xml_path = os.path.join(kept_folder_path, xml_file)
                    try:
                        tree = ET.parse(xml_path)
                        root = tree.getroot()
                        for obj in root.findall("object"):
                            name_elem = obj.find("name")
                            # Remove any -number suffix
                            name_elem.text = base_name
                        tree.write(xml_path)
                    except Exception as e:
                        print(f"Error fixing XML {xml_path}: {e}")

            # Rename the kept folder to the base brand name
            if kept_folder_path != new_folder_path:
                try:
                    os.rename(kept_folder_path, new_folder_path)
                    kept_folder_path = new_folder_path
                    print(f"Renamed folder: {folder_to_keep} -> {base_name}")
                except Exception as e:
                    print(f"Error renaming folder {kept_folder_path} -> {new_folder_path}: {e}")

            # Delete the other duplicate folders
            for folder in folders_to_delete:
                folder_path = os.path.join(category_path, folder)
                try:
                    shutil.rmtree(folder_path)
                    print(f"Deleted folder: {folder_path}")
                except Exception as e:
                    print(f"Error deleting folder {folder_path}: {e}")


if __name__ == "__main__":
    dataset_dir = "LogoDet-3K_CLEANED/LogoDet-3K"  # replace with your dataset root
    clean_duplicate_brand_folders(dataset_dir)
    print("Duplicate brand folders cleaned, renamed, and XML labels fixed.")
