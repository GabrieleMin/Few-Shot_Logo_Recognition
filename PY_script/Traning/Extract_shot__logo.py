import os
import random
from itertools import cycle

"""
##########################################################
HOW TO USE THIS CLASS:

1. Initialization:
   extract_batch = FewShotIterator(test_list, n_shot)
   --> Initializes the class with the full test_list and shot count.

2. Usage (The class instance is callable):
   task_data = extract_batch()
   --> At each call, the function returns a dictionary with the following data:

   {
       "brand_name":  selected_brand_name,   # Chosen sequentially (circularly)
       "support_set": support_set_list,      # Randomly selected images from the current brand
       "query_set":   query_set_list         # All images in 'test_list' MINUS the 'support_set'
   }

KEY BEHAVIORS:
- The brand selection iterates sequentially and restarts when finished (Circular).
- The support set is chosen randomly for that brand.
- The query set always represents the global rest of the dataset.
##########################################################
"""


class FewShotIterator:
    def __init__(self, file_list, n_shot):
        """
        Initializes the iterator class.
        It prepares the global testset and creates a cyclic iterator over the valid brands.
        """
        self.n_shot = n_shot
        
        # 1. Validation: Check if input list is empty
        if not file_list:
            raise ValueError("The test file list is empty.")

        #    (Dataset - SupportSet)  is significantly faster with sets (O(1)) compared to lists.
        self.all_files_set = set(file_list)

        # 3. Organize data by Brand
        #    We create a dictionary mapping: { 'BrandName': [list_of_image_paths] }
        self.brands_map = {}

        for file_path in file_list:
            # Extract brand name assuming structure: .../Category/Brand/Image.jpg
            brand_name = os.path.basename(os.path.dirname(file_path))

            if brand_name not in self.brands_map:
                self.brands_map[brand_name] = []
            self.brands_map[brand_name].append(file_path)

        self.valid_brands_list = list(self.brands_map.keys())

        if not self.valid_brands_list:
            raise ValueError(f"No brand found with more than {n_shot} images.")
        
        #    'itertools.cycle' creates an infinite loop over the valid brands list.
        self.brand_iterator = cycle(self.valid_brands_list)

    def __call__(self):
        """
        Executed when the class instance is called.
        Logic:
        1. Pick next brand (Sequential).
        2. Pick Support Set (Random 5 images from that brand).
        3. Pick Query Set (EVERYTHING else in the testset).
        """
        # A. Get the next brand sequentially from the cycle
        selected_brand_name = next(self.brand_iterator)

        # B. Retrieve all images specific to this chosen brand
        images_of_current_brand = self.brands_map[selected_brand_name]

        # C. Create SUPPORT SET
        #    Select 'n_shot' unique images randomly from the current brand.
        support_set_list = random.sample(images_of_current_brand, self.n_shot)

        # D. Create QUERY SET (Global Subtraction)
        #    Requirement: The Query Set must contain ALL images from the original file_list
        #    EXCEPT the ones chosen for the Support Set.
        #    Step 1: Convert support list to set for operation
        support_set_set = set(support_set_list)
        #    Step 2: Mathematical Set Difference: {All Files} - {Support Set}
        #    This leaves us with the entire dataset excluding the 5 selected images.
        query_set_set = self.all_files_set - support_set_set
        #    Step 3: Convert back to list for the return object
        query_set_list = list(query_set_set)

        return {
            "brand_name": selected_brand_name,
            "support_set": support_set_list,
            "query_set": query_set_list
        }