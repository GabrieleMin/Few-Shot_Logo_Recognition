import os
import random

def extract_few_shot_from_list(file_list, n_shot):
    """
    Receives a flat list of file paths (from the test set), reconstructs the brands,
    picks one at random, and creates a Support Set and a Query Set.
    """
    if not file_list:
        raise ValueError("The test file list is empty.")

    brands_map = {}

    for file_path in file_list:
        # Assuming structure: .../Category/Brand/Image.jpg
        # dirname -> .../Category/Brand
        # basename -> Brand
        brand_name = os.path.basename(os.path.dirname(file_path))

        if brand_name not in brands_map:
            brands_map[brand_name] = []
        brands_map[brand_name].append(file_path)

    valid_brands_list = [name for name, imgs in brands_map.items() if len(imgs) > n_shot]

    if not valid_brands_list:
        raise ValueError(f"No brand found with more than {n_shot} images.")

    # 2. Choose a random name from the list
    selected_brand_name = random.choice(valid_brands_list)

    # 3. Retrieve images using the extracted name
    all_brand_images = brands_map[selected_brand_name]

    # D. Creation of Support and Query Set (Unchanged)
    support_set = random.sample(all_brand_images, n_shot)
    query_set = list(set(all_brand_images) - set(support_set))

    return {
        "brand_name": selected_brand_name,
        "total_images": len(all_brand_images),
        "support_set": support_set,
        "query_set": query_set
    }