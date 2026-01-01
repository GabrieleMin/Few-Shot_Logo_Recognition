import matplotlib.pyplot as plt

# --- COSTANTI ---

EPOCHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Valori inizializzati a 0 pronti per essere sostituiti con i dati reali
Y_METRICS = {
    "Precision": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Recall": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "F1 Score": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Recall at Fixed Precision (R@P)": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Discriminant Ratio (J)": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Mean Average Precision (mAP)": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Contrastive Loss": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

# --- FUNZIONI ---

def plot_metrics(x_data, metrics_dict):
    for metric_name, y_values in metrics_dict.items():
        if len(y_values) != len(x_data):
            continue 

        plt.figure(figsize=(10, 6))
        
        color = 'tab:red' if 'Loss' in metric_name else 'tab:blue'
        
        plt.plot(x_data, y_values, marker='o', linestyle='-', linewidth=2, color=color, label=metric_name)
        
        plt.title(metric_name)
        plt.xlabel("Epochs")
        plt.ylabel(metric_name)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()

# --- MAIN ---

if __name__ == "__main__":
    plot_metrics(EPOCHS, Y_METRICS)