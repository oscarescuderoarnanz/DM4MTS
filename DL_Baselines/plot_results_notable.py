import json
import os
import matplotlib.pyplot as plt

def plot_roc_auc_individual(models_paths, model_names):
    all_roc_aucs = []

    for model_path in models_paths:
        roc_aucs = []

        for split in range(1, 6):
            split_path = os.path.join(model_path, f"split_{split}", "results.json")
            if os.path.exists(split_path):
                with open(split_path, "r") as f:
                    data = json.load(f)
                    roc_auc = data["metrics"]["roc_auc"]
                    roc_aucs.append(roc_auc)
            else:
                print(f"Warning: File not found {split_path}")

        all_roc_aucs.append(roc_aucs)

    # Crear el gráfico de boxplots
    fig, ax = plt.subplots(figsize=(14, 8))  # Tamaño ajustado

    plt.boxplot(
        all_roc_aucs,
        labels=model_names,
        patch_artist=True,
        boxprops=dict(color="blue", alpha=0.5),
        medianprops=dict(color="blue", linewidth=3),
        whiskerprops=dict(linestyle='-', linewidth=2, color="gray"),
        capprops=dict(linestyle='-', linewidth=2, color="gray"),
        showfliers=False
    )

    plt.ylim([0.675, 0.875])
    plt.xticks(rotation=45, ha="right", fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(axis='y')
    plt.tight_layout()

    os.makedirs("./Figures", exist_ok=True)
    plt.savefig("./Figures/roc_auc_boxplot_individual.pdf", bbox_inches="tight")
    plt.show()
    print("Gráfico guardado en ./Figures/roc_auc_boxplot_individual.pdf")


# Rutas individuales (no agrupadas)
models_paths = [
    "./E0_DL/Results_LSTM/",
    "./E0_DL/Results_GRU/",
    "./E0_DL/Results_Transformer/",
    "./E1_processing_feature_dimension/Results",
    "./E2_time_dimension_concatenation/Results",
    "./E3_hybrid_processing/Results",
    "./E4_flatten_time/Results"
]

model_names = [
    "LSTM",
    "GRU",
    "Transformer",
    "Feature-wise DR",
    "Feature-only proc.",
    "Hybrid row-column proc.",
    "Flatten MLP"
]

plot_roc_auc_individual(models_paths, model_names)
