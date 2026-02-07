import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

def generate_and_save_cm(model_name_display, model_name_file, json_path, output_dir):
    print(f"Işleniyor: {model_name_display}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'confusion_matrix' in data and 'class_names' in data:
            confusion_matrix_data = np.array(data['confusion_matrix'])
            class_names = data['class_names']
        elif 'test_results' in data and 'confusion_matrix' in data['test_results']:
             confusion_matrix_data = np.array(data['test_results']['confusion_matrix'])
             class_names = data['test_results']['class_names']
        else:
            print(f"  -> HATA: '{json_path}' içinde gerekli anahtarlar bulunamadı.")
            return

    except FileNotFoundError:
        print(f"  -> HATA: Dosya bulunamadı: {json_path}")
        return
    except Exception as e:
        print(f"  -> HATA: Dosya okunurken bir sorun oluştu: {e}")
        return

    fig, (ax_heatmap, ax_legend) = plt.subplots(
        1, 2, 
        figsize=(26, 18), 
        gridspec_kw={'width_ratios': [4, 1.5]}
    )
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

    n_classes = len(class_names)
    tick_labels = np.arange(1, n_classes + 1)
    
    sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=tick_labels, yticklabels=tick_labels,
                annot_kws={"size": 18, "weight": "bold"},
                linewidths=.5, cbar=False, ax=ax_heatmap)
    
    if "combination" in model_name_display.lower():
         title = "Confusion matrix for combination models (efficientnetb0-resnet50-regnety)"
    else:
         title = f"Confusion matrix for {model_name_display.lower()}"

    ax_heatmap.set_title(title, pad=30)
    ax_heatmap.set_ylabel('True Label')
    ax_heatmap.set_xlabel('Predicted Label')
    ax_heatmap.tick_params(axis='x', labelrotation=0, labelsize=22)
    ax_heatmap.tick_params(axis='y', labelsize=22)
    
    ax_legend.axis('off')
    
    y_pos = 1.0
    x_pos_num = -0.1
    x_pos_name = 0.05
    line_height = 0.041 

    for i, name in enumerate(class_names):
        ax_legend.text(x_pos_num, y_pos, f"{i+1}:",
                       fontsize=24,
                       fontweight='bold',
                       verticalalignment='top',
                       horizontalalignment='left',
                       fontfamily='Times New Roman')

        ax_legend.text(x_pos_name, y_pos, name,
                       fontsize=24,
                       fontstyle='italic',
                       verticalalignment='top',
                       horizontalalignment='left',
                       fontfamily='Times New Roman')
        
        y_pos -= line_height

    save_filename = f"{model_name_file}_confusion_matrix_final.png"
    save_path = os.path.join(output_dir, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  -> Grafik başarıyla kaydedildi: {save_path}\n")

output_dir = "graphs"
os.makedirs(output_dir, exist_ok=True)

models_to_plot = {
    "ResNet50": {
        "display_name": "ResNet50", 
        "file_name": "ResNet50",
        "path": "path/to/your/resnet50_logs/test_log.json"
    },
    "Combination": {
        "display_name": "Combination (EfficientNetB0-ResNet50-RegNetY)", 
        "file_name": "Combination_Model",
        "path": "path/to/your/ensemble_logs/test_log.json"
    }
}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': 'Times New Roman',
    'font.weight': 'bold',
    'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
    'axes.titlesize': 36, 
    'axes.labelsize': 30
})

if not models_to_plot or any("path/to/your" in info['path'] for info in models_to_plot.values()):
    print("HATA: 'models_to_plot' sözlüğünü kendi 'test_log.json' dosya yollarınızla güncelleyin.")
    sys.exit()

print("Karmaşıklık matrisleri oluşturuluyor...")
for model_key, model_info in models_to_plot.items():
    generate_and_save_cm(
        model_name_display=model_info['display_name'],
        model_name_file=model_info['file_name'],
        json_path=model_info['path'],
        output_dir=output_dir
    )

print("--- Tüm işlemler tamamlandı. ---")