########################################################################################################################################
# ALCUNI BRANDS SEMBRANO ESSERE DUPLICATI EX LogoDet-3K\LogoDet-3K\Leisure\stein world-1 E LogoDet-3K\LogoDet-3K\Leisure\stein world-2 #
########################################################################################################################################

from Contrastive_impl import DatasetContrastive
from dependencies import getPathsSetsByBrand, show_contrastive_with_bboxes, random, torch, transforms,SEED
random.seed(SEED)
torch.manual_seed(SEED)


 # DATASET that takes the list of files for the dataset, 
 # builds a dictionary with label -> image paths (label is taken from the image path, not from the xml because it is faster)
 # when an item from the dataset is requested it returns a triplet (anchor, positive, negative)

 # each sample has the structure: sample = {"image": img_transformed, "labels": labels_list, "bbs": bb_list}

# Returns a list of paths divided into train, validation and test.
# the directory dir is scanned to find all brands which are then split into train, validation, test so that validation and test have proportion val_split, test_split
# if total_set_size is provided only that number of images are loaded so that for each brand at least min_images_per_brand images are present


def main():
    dir = "LogoDet-3K/LogoDet-3K"
    test_split = 1/10
    val_split = 1/10

    train_data_list, val_data_list, test_data_list = getPathsSetsByBrand(dir, val_split, test_split, 1000, 15)

    print(f"images in the training set: {len(train_data_list)}")
    print(f"images in the validation set: {len(val_data_list)}")
    print(f"images in the test set: {len(test_data_list)}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = DatasetContrastive(train_data_list, transform)
    val_dataset = DatasetContrastive(val_data_list, transform)
    test_dataset = DatasetContrastive(test_data_list, transform)

    sample_idx = random.randint(0, len(train_dataset) - 1)
    img1, img2, is_positive = train_dataset[sample_idx]

    print(f"THE MATCH IS {is_positive}")

    show_contrastive_with_bboxes(img1, img2)



    return

if __name__ == "__main__":
    main()
