import torch
import torch.nn.functional as F

class MetricEvaluator:
    """
    A class to calculate evaluation metrics for Few-Shot Learning and Metric Learning.
    
    Implements:
    1. Discriminant Ratio (J): Optimized scalar implementation (O(d) memory).
    2. Mean Average Precision (mAP): Ranking quality metric.
    3. Recall at Fixed Precision (R@P): Operational metric.
    4. Precision & Recall: Raw metrics at a specific similarity threshold.
    5. F1 Score: Harmonic mean of Precision and Recall.
    """

    def __init__(self, device=None):
        """
        Initialize the evaluator.
        
        Args:
            device (str): 'cuda' or 'cpu'. If None, detects automatically.
        """
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        self.epsilon = 1e-6  # For numerical stability

    def compute_discriminant_ratio(self, embeddings, labels):
        """
        Calculates the Discriminant Ratio (J) using the optimized Scalar approach.
        
        Theory:
            J = Tr(Sb) / Tr(Sw)
            Using the Trace Trick: Tr(Sw) = Tr(St) - Tr(Sb)
            
        Args:
            embeddings (torch.Tensor): Tensor of shape (Batch_Size, Dimension).
            labels (torch.Tensor): Tensor of class labels.
            
        Returns:
            float: The Discriminant Ratio score.
        """
        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)
        
        # 1. Global Mean Computation
        global_mean = embeddings.mean(dim=0)
        
        # 2. Calculate Trace of Total Scatter (St)
        # Sum of squared Euclidean distances of all points from the global mean.
        tr_st = torch.sum((embeddings - global_mean) ** 2)
        
        # 3. Calculate Trace of Between-Class Scatter (Sb)
        tr_sb = 0
        unique_classes = torch.unique(labels)
        
        for c in unique_classes:
            class_mask = (labels == c)
            class_embeddings = embeddings[class_mask]
            n_c = class_embeddings.size(0) 
            
            if n_c > 0:
                mu_c = class_embeddings.mean(dim=0)
                tr_sb += n_c * torch.sum((mu_c - global_mean) ** 2)
        
        # 4. Calculate Trace of Within-Class Scatter (Sw)
        tr_sw = tr_st - tr_sb
        
        # Calculate J
        j_score = tr_sb / (tr_sw + self.epsilon)
        
        return j_score.item()

    def compute_map(self, query_emb, gallery_emb, query_labels, gallery_labels):
        """
        Calculates Mean Average Precision (mAP).
        """
        query_emb = query_emb.to(self.device)
        gallery_emb = gallery_emb.to(self.device)
        query_labels = query_labels.to(self.device)
        gallery_labels = gallery_labels.to(self.device)

        # L2 Normalize for Cosine Similarity
        query_emb = F.normalize(query_emb, p=2, dim=1)
        gallery_emb = F.normalize(gallery_emb, p=2, dim=1)
        
        # Similarity Matrix: S = Q * G^T
        similarity_matrix = torch.matmul(query_emb, gallery_emb.T)
        
        num_queries = query_labels.size(0)
        average_precisions = []

        for i in range(num_queries):
            scores = similarity_matrix[i]
            target_label = query_labels[i]
            
            # Ranking
            sorted_indices = torch.argsort(scores, descending=True)
            sorted_gallery_labels = gallery_labels[sorted_indices]
            
            # Relevance Mask
            relevance_mask = (sorted_gallery_labels == target_label).float()
            
            total_relevant = relevance_mask.sum()
            if total_relevant == 0:
                average_precisions.append(0.0)
                continue
            
            # Cumulative Precision
            cumsum = torch.cumsum(relevance_mask, dim=0)
            ranks = torch.arange(1, len(relevance_mask) + 1).to(self.device)
            precisions = cumsum / ranks
            
            # Average Precision (AP)
            ap = (precisions * relevance_mask).sum() / total_relevant
            average_precisions.append(ap.item())
            
        if not average_precisions:
            return 0.0
        return sum(average_precisions) / len(average_precisions)

    def compute_precision_recall(self, similarity_scores, is_match, threshold=0.5):
        """
        Calculates raw Precision and Recall at a specific similarity threshold.
        
        Definitions:
            Precision = TP / (TP + FP)
            Recall    = TP / (TP + FN)
        
        Args:
            similarity_scores (torch.Tensor): 1D tensor of scores (0.0 to 1.0).
            is_match (torch.Tensor): 1D binary tensor (Ground Truth).
            threshold (float): Cutoff for deciding if a retrieval is Positive.
            
        Returns:
            tuple: (precision, recall)
        """
        similarity_scores = similarity_scores.to(self.device)
        is_match = is_match.to(self.device)
        
        # Binarize predictions: 1 if score >= threshold (Positive), else 0 (Negative)
        predicted_positive = (similarity_scores >= threshold).float()
        
        # True Positives (TP): Predicted Positive AND Actually Match
        tp = (predicted_positive * is_match).sum()
        
        # False Positives (FP): Predicted Positive BUT Actually Non-Match
        fp = (predicted_positive * (1 - is_match)).sum()
        
        # False Negatives (FN): Predicted Negative BUT Actually Match
        # (We invert the prediction mask to find negatives)
        fn = ((1 - predicted_positive) * is_match).sum()
        
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        
        return precision.item(), recall.item()

    def compute_recall_at_fixed_precision(self, similarity_scores, is_match, min_precision=0.95):
        """
        Calculates Recall at a Fixed Precision (R@P).
        Finds the lowest threshold where Precision >= min_precision.
        """
        similarity_scores = similarity_scores.to(self.device)
        is_match = is_match.to(self.device)

        sorted_indices = torch.argsort(similarity_scores, descending=True)
        sorted_matches = is_match[sorted_indices]
        
        tps = torch.cumsum(sorted_matches, dim=0)
        total_retrieved = torch.arange(1, len(sorted_matches) + 1).to(self.device)
        
        precisions = tps / total_retrieved
        
        # Find indices where Precision satisfies the constraint
        valid_indices = torch.where(precisions >= min_precision)[0]
        
        if len(valid_indices) == 0:
            return 0.0
            
        cutoff_index = valid_indices[-1]
        
        # Recall = TP_at_cutoff / Total_Relevant_In_Dataset
        total_relevant_in_dataset = is_match.sum()
        
        if total_relevant_in_dataset == 0:
            return 0.0
            
        recall = tps[cutoff_index] / total_relevant_in_dataset
        
        return recall.item()

    def compute_f1_score(self, precision, recall):
        """
        Calculates F1 Score (Harmonic Mean).
        """
        if (precision + recall) == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


# ==========================================
# MAIN FOR TESTING
# ==========================================
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
    q_norm = F.normalize(q_emb, p=2, dim=1)
    g_norm = F.normalize(g_emb, p=2, dim=1)
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