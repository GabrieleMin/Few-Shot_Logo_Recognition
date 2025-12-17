# TO DO:
# review code
# Implement different similarity funcitons

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Extract_shot__logo import FewShotIterator
from Metric_central_implementation import MetricEvaluator
from my_dataset import getPathsSetsByBrand  # function for train/val/test split
from my_dataset import DatasetTest  # function for train/val/test split

# ============================
# CONFIG
# ============================
SEED = 101
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EMBEDDING_DIM = 128          # adjust if your model has a different embedding size
MODEL_PATH = "model/final_model.pth"
DATA_ROOT = "../LogoDet-3K/LogoDet-3K"
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
N_SHOT = 1
NUM_EPISODES = 100
PRED_THREASHOLD = 0.5


from torchvision import transforms

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# Will load the model from the path specified
def load_model(model_path, device):
    """Replace YourEmbeddingModel with your actual model class."""
    from Implementation_ResNet50 import LogoResNet50

    model = LogoResNet50(embedding_dim=EMBEDDING_DIM)
    # state = torch.load(model_path)
    # model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# computes the cosine similarity between an embedings and a list of query embedings
def cosine_similarity(averaged_support_embeddings, query_embeddings_tensor):

    # Normalize embeddings if you want cosine similarity
    support_emb_norm = F.normalize(averaged_support_embeddings, p=2, dim=0)       # [embedding_dim]
    query_emb_norm = F.normalize(query_embeddings_tensor, p=2, dim=1)             # [num_queries, embedding_dim]
    
    # Compute cosine similarity
    sims = torch.matmul(query_emb_norm, support_emb_norm)  # [num_queries]
    return sims

# performs the actual few shot cycles
def evaluate_few_shot(model, fewshot_iterator, transform, device, num_episodes=100):
    evaluator = MetricEvaluator(device=device)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    r_at_95p = []

    retrieval_stats = []  # NEW: list of (num_correct, num_total) per episode

    for _ in range(num_episodes):
        task = fewshot_iterator()
        support_paths = task["support_set"]
        query_paths = task["query_set"]

        # Build datasets and loaders
        support_dataset = DatasetTest(support_paths, transform)
        query_dataset = DatasetTest(query_paths, transform)
        
        support_loader = DataLoader(support_dataset, batch_size=32)
        query_loader = DataLoader(query_dataset, batch_size=64)

        # Extract embeddings
        support_embeddings = []
        query_embeddings = []
        query_labels = []

        # Compute embeddings for support set 
        for data in support_loader:
            images = data["image"].to(device)
            support_embeddings.append(model(images))

            batch_labels = data["label"]
            support_brand = batch_labels[0]

        support_embeddings_tensor = torch.cat(support_embeddings)

        # Average embeddings
        averaged_support_embeddings = support_embeddings_tensor.mean(dim=0)
        
        # Compute embeddings for query set
        for data in query_loader:
            images = data["image"].to(device)
            query_embeddings.append(model(images))

            batch_labels = data["label"]
            query_labels.append(batch_labels)

        # query_embeddings and query_labels are list of tensors, this unrolls them
        query_embeddings_tensor = torch.cat(query_embeddings)
        query_labels_tensor = torch.cat(query_labels)

        # Compute similarity
        sims = cosine_similarity(averaged_support_embeddings, query_embeddings_tensor)

        # Ground truth: query belongs to support brand?
        gt = (query_labels_tensor == support_brand).float()

        # Predictions, does the model predict it is the same brand?
        pred = (sims >= PRED_THREASHOLD).float().cpu()

        # Accuracy
        acc = (pred == gt).float().mean().item()
        accuracies.append(acc)

        # Precision, Recall, F1
        prec, rec = evaluator.compute_precision_recall(sims, gt, threshold=PRED_THREASHOLD)
        f1 = evaluator.compute_f1_score(prec, rec)
        r95 = evaluator.compute_recall_at_fixed_precision(sims, gt, min_precision=0.95)

        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        r_at_95p.append(r95)

    # Aggregate results
    results = {
        "accuracy": sum(accuracies) / len(accuracies),
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "f1": sum(f1_scores) / len(f1_scores),
        "r@95p": sum(r_at_95p) / len(r_at_95p),
    }
    return results


def main():
    torch.manual_seed(SEED)

    print("=== FEW-SHOT MODEL EVALUATION ===")

    # Load test set
    _, _, test_data_list = getPathsSetsByBrand(
        DATA_ROOT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        total_set_size=5000,
        min_images_per_brand=6
    )
    print(f"Test images: {len(test_data_list)}")

    # Load model
    model = load_model(MODEL_PATH, DEVICE)

    # Few-shot iterator
    fewshot = FewShotIterator(test_data_list, n_shot=N_SHOT)

    # Evaluate
    results = evaluate_few_shot(
        model,
        fewshot,
        transform,
        DEVICE,
        num_episodes=NUM_EPISODES
    )

    print("\n=== Few-Shot Evaluation Results ===")
    for k, v in results.items():
        print(f"{k.capitalize():<10}: {v:.4f}")


    print("\n=== Evaluation Completed ===")

if __name__ == "__main__":
    main()
