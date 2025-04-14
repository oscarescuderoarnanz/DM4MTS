import json
import os
import matplotlib.pyplot as plt
from matplotlib.table import Table

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

    # Crear el gráfico de boxplots con figura más ancha
    fig, ax = plt.subplots(figsize=(14, 10))  # Aumentado ancho

    plt.boxplot(all_roc_aucs, labels=[str(i+1) for i in range(len(model_names))], patch_artist=True,
                boxprops=dict(color="blue", alpha=0.5),
                medianprops=dict(color="blue", linewidth=3),
                whiskerprops=dict(linestyle='-', linewidth=2, color="gray"),
                capprops=dict(linestyle='-', linewidth=2, color="gray"),
                showfliers=False)

    plt.ylim([0.675, 0.875])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis='y')

    # Tabla más centrada y espaciosa
    table_data = [[f"{i+1}", name] for i, name in enumerate(model_names)]
    ax_table = plt.gcf().add_subplot(111)
    ax_table.axis('off')

    # Ajustar el bbox (posición y tamaño) y el ancho de columnas
    table = Table(ax_table, bbox=[0.05, -0.65, 0.9, 0.4])  # bbox: [left, bottom, width, height]
    table.auto_set_font_size(False)

    for row, (idx, name) in enumerate(table_data):
        # Ajustar anchos: más espacio para la columna de texto
        cell_id = table.add_cell(row, 0, width=0.1, height=0.3, text=idx, loc='center', facecolor='lightgray', edgecolor='black')
        cell_id.get_text().set_fontsize(17)

        cell_name = table.add_cell(row, 1, width=0.9, height=0.3, text=name, loc='center', facecolor='white', edgecolor='black')
        cell_name.get_text().set_fontsize(17)

    # Encabezados de tabla
    cell_header_id = table.add_cell(-1, 0, width=0.1, height=0.35, text="ID", loc='center', facecolor='lightblue', edgecolor='black')
    cell_header_id.get_text().set_fontsize(18)

    cell_header_name = table.add_cell(-1, 1, width=0.9, height=0.35, text="Model Description", loc='center', facecolor='lightblue', edgecolor='black')
    cell_header_name.get_text().set_fontsize(18)

    ax_table.add_table(table)

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
    "Raw MTS - LSTM",
    "Raw MTS - GRU",
    "Raw MTS - Transformer",
    "DL with feature processing",
    "DL with time processing",
    "Hybrid ML row-column processing",
    "ML vectorizing time dimension"
]

plot_roc_auc_individual(models_paths, model_names)
