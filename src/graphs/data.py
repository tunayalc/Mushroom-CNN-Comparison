import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

data_and_output_dir = "graphs"
input_filename = os.path.join(data_and_output_dir, 'significant_epochs_data.json')
os.makedirs(data_and_output_dir, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': 'Times New Roman',
    'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
    'axes.titlesize': 22, 'axes.labelsize': 18, 'xtick.labelsize': 14,
    'ytick.labelsize': 14, 'legend.fontsize': 12, 'legend.title_fontsize': 14,
    'axes.spines.top': False, 'axes.spines.right': False,
})

try:
    with open(input_filename, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
    print(f"'{input_filename}' başarıyla yüklendi. Grafikler oluşturuluyor...\n")
except FileNotFoundError:
    print(f"HATA: '{input_filename}' dosyası bulunamadı. Lütfen önce 'data.py' scriptini çalıştırın.")
    sys.exit()

processed_data.pop('meta', {})
colors = plt.cm.get_cmap('tab10', len(processed_data))

print("Oluşturuluyor: 1. Training Accuracy Comparison Plot")
plt.figure(figsize=(15, 9))
for i, (model_name, data) in enumerate(processed_data.items()):
    plot_data = data['train_accuracy']
    best_acc = plot_data['best_value'] * 100
    legend_label = f'{model_name}: {best_acc:.0f}%'
    plot_values_percent = [0] + [v * 100 for v in plot_data['values']]
    new_epoch_axis = range(len(plot_values_percent))
    plt.plot(new_epoch_axis, plot_values_percent, label=legend_label, linewidth=2.5, color=colors(i))
plt.title('Model training accuracy per epoch', pad=20); plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)')
plt.ylim(0, 101); plt.xlim(0, 20)
xticks_locs = np.arange(0, 21, 2); xtick_labels = [str(x) if x != 0 else '' for x in xticks_locs]
plt.xticks(ticks=xticks_locs, labels=xtick_labels); plt.yticks(np.arange(0, 101, 10))
plt.legend(title='Models', loc='lower right', frameon=True, framealpha=1, edgecolor='black', facecolor='white'); plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
save_path = os.path.join(data_and_output_dir, 'comparison_training_accuracy.png')
plt.savefig(save_path, dpi=300); plt.close()
print(f"  -> Kaydedildi: {save_path}\n")

print("Oluşturuluyor: 2. Validation Accuracy Comparison Plot")
plt.figure(figsize=(15, 9))
for i, (model_name, data) in enumerate(processed_data.items()):
    plot_data = data['val_accuracy']
    best_acc = plot_data['best_value'] * 100
    legend_label = f'{model_name}: {best_acc:.0f}%'
    plot_values_percent = [0] + [v * 100 for v in plot_data['values']]
    new_epoch_axis = range(len(plot_values_percent))
    plt.plot(new_epoch_axis, plot_values_percent, label=legend_label, linewidth=2.5, color=colors(i))
plt.title('Model validation accuracy per epoch', pad=20); plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)')
plt.ylim(0, 101); plt.xlim(0, 20)
xticks_locs = np.arange(0, 21, 2); xtick_labels = [str(x) if x != 0 else '' for x in xticks_locs]
plt.xticks(ticks=xticks_locs, labels=xtick_labels); plt.yticks(np.arange(0, 101, 10))
plt.legend(title='Models', loc='lower right', frameon=True, framealpha=1, edgecolor='black', facecolor='white'); plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
save_path = os.path.join(data_and_output_dir, 'comparison_validation_accuracy.png')
plt.savefig(save_path, dpi=300); plt.close()
print(f"  -> Kaydedildi: {save_path}\n")

print("Oluşturuluyor: 3. Training Loss Comparison Plot")
plt.figure(figsize=(15, 9))
for i, (model_name, data) in enumerate(processed_data.items()):
    plot_data = data['train_loss']
    best_loss = plot_data['best_value']
    legend_label = f'{model_name}: {best_loss:.4f}'
    plot_values = [0] + plot_data['values']
    new_epoch_axis = range(len(plot_values))
    plt.plot(new_epoch_axis, plot_values, label=legend_label, linewidth=2.5, color=colors(i))
plt.title('Model training loss per epoch', pad=20); plt.xlabel('Epochs'); plt.ylabel('Loss')
plt.xlim(0, 20); plt.ylim(bottom=0)
xticks_locs = np.arange(0, 21, 2); xtick_labels = [str(x) if x != 0 else '' for x in xticks_locs]
plt.xticks(ticks=xticks_locs, labels=xtick_labels); plt.yticks()
plt.legend(title='Models', loc='upper right', frameon=True, framealpha=1, edgecolor='black', facecolor='white'); plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
save_path = os.path.join(data_and_output_dir, 'comparison_training_loss.png')
plt.savefig(save_path, dpi=300); plt.close()
print(f"  -> Kaydedildi: {save_path}\n")

print("Oluşturuluyor: 4. Validation Loss Comparison Plot")
plt.figure(figsize=(15, 9))
for i, (model_name, data) in enumerate(processed_data.items()):
    plot_data = data['val_loss']
    best_loss = plot_data['best_value']
    legend_label = f'{model_name}: {best_loss:.4f}'
    plot_values = [0] + plot_data['values']
    new_epoch_axis = range(len(plot_values))
    plt.plot(new_epoch_axis, plot_values, label=legend_label, linewidth=2.5, color=colors(i))
plt.title('Model validation loss per epoch', pad=20); plt.xlabel('Epochs'); plt.ylabel('Loss')
plt.xlim(0, 20); plt.ylim(bottom=0)
xticks_locs = np.arange(0, 21, 2); xtick_labels = [str(x) if x != 0 else '' for x in xticks_locs]
plt.xticks(ticks=xticks_locs, labels=xtick_labels); plt.yticks()
plt.legend(title='Models', loc='upper right', frameon=True, framealpha=1, edgecolor='black', facecolor='white'); plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
save_path = os.path.join(data_and_output_dir, 'comparison_validation_loss.png')
plt.savefig(save_path, dpi=300); plt.close()
print(f"  -> Kaydedildi: {save_path}\n")

print("--- TÜM GRAFİKLER BAŞARIYLA OLUŞTURULDU ---")