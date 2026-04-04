import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split
import sys

# Lütfen ana veri setinizin bulunduğu klasörün yolunu buraya girin.
# Örnek: "data/raw_images" veya "C:/Users/User/Desktop/dataset"
dataset_path = "path/to/your/dataset_folder"

output_dir = "split_dataset"
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

if not os.path.exists(dataset_path):
    print(f"HATA: Belirtilen '{dataset_path}' yolu bulunamadı.")
    print("Lütfen 'split_code.py' dosyasındaki 'dataset_path' değişkenini doğru şekilde güncelleyin.")
    sys.exit()

if os.path.exists(output_dir):
    print(f"'{output_dir}' klasörü zaten var. İçeriği temizlenip yeniden oluşturuluyor...")
    shutil.rmtree(output_dir)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
print(f"'{output_dir}' klasörü başarıyla oluşturuldu.")

image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.jfif", "*.JPG", "*.PNG"]
images = []
labels = []

class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
labels_dict = {name: i for i, name in enumerate(class_names)}

for class_name, label_idx in labels_dict.items():
    class_path = os.path.join(dataset_path, class_name)
    found_images = []
    for ext in image_extensions:
        found_images.extend(glob(os.path.join(class_path, ext)))
    
    images.extend(found_images)
    labels.extend([label_idx] * len(found_images))
    print(f"- Sınıf: '{class_name}', Bulunan görsel sayısı: {len(found_images)}")

print("-" * 30)
print(f"Toplam sınıf sayısı: {len(class_names)}")
print(f"Toplam görsel sayısı: {len(images)}")

if len(images) == 0:
    raise ValueError(f"'{dataset_path}' klasöründe hiç görsel bulunamadı! Lütfen yolu kontrol edin.")

X_temp, X_test, y_temp, y_test = train_test_split(
    images, labels, test_size=0.15, stratify=labels, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42
)

print("-" * 30)
print(f"Eğitim seti boyutu: {len(X_train)}")
print(f"Doğrulama seti boyutu: {len(X_val)}")
print(f"Test seti boyutu: {len(X_test)}")
print("-" * 30)

def copy_files(file_list, label_list, destination_dir):
    for img_path, label in zip(file_list, label_list):
        class_name = class_names[label]
        target_folder = os.path.join(destination_dir, class_name)
        os.makedirs(target_folder, exist_ok=True)
        shutil.copy(img_path, target_folder)

print("Dosyalar kopyalanıyor...")
copy_files(X_train, y_train, train_dir)
copy_files(X_val, y_val, val_dir)
copy_files(X_test, y_test, test_dir)

print("\n\u2705 Veri ayırma ve kopyalama işlemi başarıyla tamamlandı!")