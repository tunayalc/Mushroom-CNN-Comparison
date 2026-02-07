import json
import numpy as np
import os
import sys

def select_final_epochs(metric_values, val_accuracy_values, num_points=20):
    total_epochs = len(metric_values)
    if total_epochs <= num_points:
        return list(range(1, total_epochs + 1)), metric_values

    best_val_epoch_index = int(np.argmax(val_accuracy_values))
    first_index = 0
    last_index = total_epochs - 1
    mandatory_indices = sorted(list(set([first_index, best_val_epoch_index, last_index])))
    
    diffs = np.abs(np.diff(np.array(metric_values)))
    initial_diff = np.abs(metric_values[0])
    full_diffs = np.insert(diffs, 0, initial_diff)
    indexed_diffs = list(enumerate(full_diffs))
    
    remaining_diffs = [item for item in indexed_diffs if item[0] not in mandatory_indices]
    remaining_diffs.sort(key=lambda x: x[1], reverse=True)
    
    num_needed = num_points - len(mandatory_indices)
    additional_indices = [index for index, diff in remaining_diffs[:num_needed]]
    
    final_indices = sorted(mandatory_indices + additional_indices)
    
    final_values = [metric_values[i] for i in final_indices]
    final_epoch_numbers = [i + 1 for i in final_indices]
    
    return final_epoch_numbers, final_values

model_log_files = {
    'DenseNet121': 'path/to/your/densenet121/train_log.json',
    'Ensemble (EfficientNetB0-DenseNet121)': 'path/to/your/ensemble_eff_dense/train_log.json',
    'ConvNeXt-Tiny': 'path/to/your/logs_convnext_tiny/train_log.json',
    'EfficientNetB0': 'path/to/your/logs_efficientnetb0/train_log.json',
    'MobileNetV3-Large': 'path/to/your/logs_mobilenetv3/train_log.json',
    'Ensemble (MobileNetV3-ConvNeXt)': 'path/to/your/mobilenet_convnext_ensemble/train_log.json',
    'Ensemble (RegNetY-ResNet50)': 'path/to/your/regnet_resnet_ensemble/train_log.json',
    'RegNetY': 'path/to/your/regnet/train_log.json',
    'ResNet50': 'path/to/your/resnet50/train_log.json',
}

output_data_dir = "graphs"
os.makedirs(output_data_dir, exist_ok=True)

if not model_log_files or any("path/to/your" in path for path in model_log_files.values()):
    print("HATA: 'model_log_files' sözlüğünü kendi 'train_log.json' dosya yollarınızla güncelleyin.")
    sys.exit()

processed_data = {}
print("Veri işleme başladı...")
for model_name, file_path in model_log_files.items():
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        processed_data[model_name] = {}
        metrics_to_process = ['train_accuracy', 'val_accuracy', 'train_loss', 'val_loss']
        val_accuracy_full = log_data['val_accuracy']

        for metric in metrics_to_process:
            full_list = log_data[metric]
            epoch_numbers, values = select_final_epochs(full_list, val_accuracy_full, num_points=20)
            
            if 'accuracy' in metric:
                best_value = float(max(full_list))
            else:
                best_value = float(min(full_list))
            
            processed_data[model_name][metric] = {
                'epochs': epoch_numbers,
                'values': values,
                'best_value': best_value
            }
        print(f" -> {model_name} başarıyla işlendi.")
    except Exception as e:
        print(f" !!! HATA: {model_name} işlenirken hata oluştu: {e}")

output_filename = os.path.join(output_data_dir, 'significant_epochs_data.json')
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, indent=4)
print(f"\nVeri işleme tamamlandı. Sonuçlar '{output_filename}' dosyasına kaydedildi.")