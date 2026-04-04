# Mushroom-CNN-Comparison

Ankara Üniversitesi Yapay Zeka Enstitüsü'ndeki akademik görüntü işleme çalışmasında kullanılan model karşılaştırma kod tabanı

## Genel Bakış

`Mushroom-CNN-Comparison`, mantar görselleri üzerinde farklı CNN mimarilerinin performansını karşılaştırmak amacıyla yürüttüğümüz akademik çalışmanın kaynak kodlarını içeriyor. Proje tek model eğitmekten çok, farklı mimarilerin davranışını karşılaştırmalı biçimde değerlendirmeye odaklanıyor.

## Karşılaştırılan Mimariler

- ConvNeXt
- EfficientNetB0
- MobileNetV3

## Çalışmanın Kapsamı

| Alan | İçerik |
| --- | --- |
| Model Eğitimi | farklı CNN mimarilerinin eğitimi |
| Deney Takibi | eğitim logları ve epoch davranışı |
| Değerlendirme | confusion matrix ve metrik analizi |
| Görselleştirme | grafik üretimi ve sonuç karşılaştırmaları |

## Klasör Yapısı

### `src/models/`

Model eğitimine odaklanan ana bölüm. Farklı CNN mimarileri için ayrı script'ler ve eğitim yardımcıları burada yer alıyor.

### `src/graphs/`

Confusion matrix, eğitim grafikleri ve sonuç karşılaştırmalarını hazırlayan analiz katmanı.

## Repo Yapısı

```text
Mushroom-CNN-Comparison/
|-- src/
|   |-- graphs/
|   `-- models/
|-- requirements.txt
`-- README.md
```

## Kullanılan Teknolojiler

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
- görüntü işleme ve sınıflandırma
