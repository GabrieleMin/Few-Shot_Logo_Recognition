from Metric_central_implementation import MetricEvaluator
import torch.nn.functional as funcF
import torch
if __name__ == "__main__":
    print("=== Initializing Metric Evaluator Test ===")
    
    # Setup
    device = 'cpu' 
    evaluator = MetricEvaluator(device=device)
    
    # Generate Synthetic Data
    print("-> Generating synthetic embeddings...")
    torch.manual_seed(42)
    
    num_classes = 3
    dim = 128
    samples_per_class = 50
    centroids = torch.randn(num_classes, dim) * 5 
    
    embeddings_list = []
    labels_list = []
    
    for c in range(num_classes):
        noise = torch.randn(samples_per_class, dim)
        class_data = centroids[c] + noise
        embeddings_list.append(class_data)
        labels_list.append(torch.full((samples_per_class,), c))
        
    embeddings = torch.cat(embeddings_list)
    labels = torch.cat(labels_list)
    
    # -------------------------------------------------
    # TEST 1: Discriminant Ratio (J)
    # -------------------------------------------------
    print("\n--- Test 1: Discriminant Ratio (J) ---")
    j_score = evaluator.compute_discriminant_ratio(embeddings, labels)
    print(f"   Calculated J Score: {j_score:.4f}")

    # -------------------------------------------------
    # TEST 2: Mean Average Precision (mAP)
    # -------------------------------------------------
    print("\n--- Test 2: Mean Average Precision (mAP) ---")
    query_indices = []
    gallery_indices = []
    
    for c in range(num_classes):
        start = c * samples_per_class
        query_indices.extend(range(start, start + 5))
        gallery_indices.extend(range(start + 5, start + samples_per_class))
        
    q_emb = embeddings[query_indices]
    q_lbl = labels[query_indices]
    g_emb = embeddings[gallery_indices]
    g_lbl = labels[gallery_indices]
    
    map_score = evaluator.compute_map(q_emb, g_emb, q_lbl, g_lbl)
    print(f"   Calculated mAP: {map_score:.4f}")
    
    # -------------------------------------------------
    # PREPARE DATA FOR TESTS 3, 4, 5 (Pairwise)
    # -------------------------------------------------
    q_norm = funcF.normalize(q_emb, p=2, dim=1)
    g_norm = funcF.normalize(g_emb, p=2, dim=1)
    sim_matrix = torch.matmul(q_norm, g_norm.T)
    scores_flat = sim_matrix.view(-1)
    matches_matrix = (q_lbl.unsqueeze(1) == g_lbl.unsqueeze(0)).float()
    matches_flat = matches_matrix.view(-1)
    
    # -------------------------------------------------
    # TEST 3: Raw Precision and Recall
    # -------------------------------------------------
    print("\n--- Test 3: Raw Precision & Recall (at threshold 0.5) ---")
    threshold = 0.5
    prec, rec = evaluator.compute_precision_recall(scores_flat, matches_flat, threshold=threshold)
    print(f"   Threshold: {threshold}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")

    # -------------------------------------------------
    # TEST 4: Recall @ Fixed Precision
    # -------------------------------------------------
    print("\n--- Test 4: Recall at Fixed Precision (R@P) ---")
    target_precision = 0.95
    recall_at_p = evaluator.compute_recall_at_fixed_precision(
        scores_flat, matches_flat, min_precision=target_precision
    )
    print(f"   Target Precision: {target_precision}")
    print(f"   Recall achieved:  {recall_at_p:.4f}")
    
    # -------------------------------------------------
    # TEST 5: F1 Score
    # -------------------------------------------------
    print("\n--- Test 5: F1 Score ---")
    # Calculating F1 based on the Raw metrics from Test 3
    f1 = evaluator.compute_f1_score(prec, rec)
    print(f"   F1 Score (from Test 3 metrics): {f1:.4f}")

    print("\n=== Test Completed Successfully ===")