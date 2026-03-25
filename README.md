# Mushroom-CNN-Comparison

Bu proje, mantar siniflandirma problemi icin farkli CNN mimarilerini karsilastirir ve model performansini hem sayisal hem gorsel raporlarla analiz eder.

## Hızlı Başlangıç
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Model egitimi ve loglama adimlarini tamamladiktan sonra analiz scriptlerini calistirin.

## Yöntem
- Birden fazla CNN mimarisinin ayni veri uzerinde egitilmesi
- Egitim/test loglarinin JSON formatinda kaydedilmesi
- Confusion matrix ve performans grafikleri ile karsilastirmali analiz

## Sonuçlar
Sonuclar veri bolme stratejisi, augmentasyon ve hiperparametrelere gore degisir. Karsilastirma yaparken ayni split/seed ile tekrarlanmasi onerilir.

## Proje Yapısı
- `src/models/`: model tanimlari ve egitim scriptleri
- `src/graphs/`: analiz ve gorsellestirme scriptleri
- `train_log.json`, `test_log.json`: deney ciktilari

## Ana Dosyalar
- `src/models/` altindaki egitim kodlari
- `src/graphs/` altindaki raporlama kodlari
