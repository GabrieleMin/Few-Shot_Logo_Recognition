
import matplotlib.pyplot as plt
import os

# --- COSTANTI ---

EPOCHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Valori inizializzati a 0 (dati placeholder)
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

def save_plot_metrics(x_data, metrics_dict, output_folder="plt_result_metric"):
    """
    Genera i plot per ogni metrica e li salva nella cartella specificata.
    """
    # Crea la cartella se non esiste
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Cartella creata: {output_folder}")

    for metric_name, y_values in metrics_dict.items():
        if len(y_values) != len(x_data):
            print(f"Attenzione: Lunghezza dati non corrispondente per {metric_name}, salto il plot.")
            continue 

        plt.figure(figsize=(10, 6))
        
        # Selezione colore
        color = 'tab:red' if 'Loss' in metric_name else 'tab:blue'
        
        # Creazione plot
        plt.plot(x_data, y_values, marker='o', linestyle='-', linewidth=2, color=color, label=metric_name)
        
        plt.title(metric_name)
        plt.xlabel("Epochs")
        plt.ylabel(metric_name)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Pulizia del nome per il file (rimuove spazi e parentesi)
        safe_filename = metric_name.replace(" ", "_").replace("(", "").replace(")", "") + ".png"
        save_path = os.path.join(output_folder, safe_filename)
        
        # Salvataggio e chiusura figura
        plt.savefig(save_path)
        plt.close() # Importante per liberare memoria
        
        print(f"Salvato: {save_path}")

# --- MAIN ---

if __name__ == "__main__":
    save_plot_metrics(EPOCHS, Y_METRICS)