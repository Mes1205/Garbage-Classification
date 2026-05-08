# 🗑️ Garbage Classification CNN

Proyek klasifikasi gambar sampah menggunakan Convolutional Neural Network (CNN) dengan TensorFlow/Keras.

## 📊 Dataset

**Garbage Classification Dataset** dari Kaggle  
Link: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

| Kelas       | Jumlah Gambar |
|-------------|---------------|
| Glass       | ~501          |
| Paper       | ~594          |
| Cardboard   | ~403          |
| Plastic     | ~482          |
| Metal       | ~410          |
| Trash       | ~137          |
| **Total**   | **~2527**     |

### Split Dataset
| Set        | Rasio | Keterangan             |
|------------|-------|------------------------|
| Train      | 70%   | Data pelatihan         |
| Validation | 10%   | Data validasi          |
| Test       | 20%   | Data pengujian akhir   |

## 🏗️ Arsitektur Model

Transfer learning berbasis EfficientNetV2S (pretrained ImageNet) sebagai backbone (base frozen di fase awal), ditambahkan head:
- GlobalAveragePooling2D
- Dense(512) → BatchNormalization → Dropout(0.4)
- Dense(256) → Dropout(0.3)
- Dense(6, softmax)

Base model awalnya frozen; kemudian selective fine-tuning (unfreeze ~100 layer terakhir).

## ⚙️ Hyperparameter & Training Strategy

- Image size: 224 × 224
- Batch size: 32
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Augmentasi: ImageDataGenerator (rescale=1/255 + transformasi untuk train)

Training:
- Fase 1: Train head (base frozen) — 15 epochs, LR = 1e-3
- Fase 2: Fine-tune (selective unfreeze) — hingga 80 epochs dengan EarlyStopping, LR = 1e-5, ReduceLROnPlateau aktif

(Class weights dihitung otomatis untuk menangani imbalance.)

## 📈 Evaluasi & Eksport

- Evaluasi: training / validation / test, confusion matrix & classification report ditampilkan di notebook.
- Penyimpanan model:
  - SavedModel: gunakan model.save('saved_model') (notebook sudah diperbarui)
  - TF-Lite: ./tflite/model.tflite
  - TF.js: ./tfjs_model/ (konversi via tensorflowjs_converter)

## 🔍 Inference Contoh

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model('saved_model')

# Load & preprocess gambar
img = image.load_img('gambar_sampah.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediksi
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
pred = model.predict(img_array)
print(f"Prediksi: {classes[np.argmax(pred)]} ({np.max(pred)*100:.2f}%)")

## 📈 Hasil

| Set        | Accuracy | Loss  |
|------------|----------|-------|
| Train      | ≥ 85%    | -     |
| Validation | ≥ 85%    | -     |
| Test       | ≥ 85%    | -     |

## 💾 Format Model Tersimpan

- **SavedModel** (`./saved_model/`) — Deployment TensorFlow/server
- **TF-Lite** (`./tflite/model.tflite`) — Mobile & embedded devices
- **TF.js** (`./tfjs_model/`) — Browser & JavaScript apps

## 🚀 Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Siapkan dataset
Unduh dataset dari Kaggle dan ekstrak sehingga strukturnya:
```
Garbage classification/
    Garbage classification/
        cardboard/
        glass/
        metal/
        paper/
        plastic/
        trash/
```

### 3. Jalankan notebook
```bash
jupyter notebook notebook.ipynb
```

### 4. (Opsional) Install tensorflowjs untuk konversi TFJS
```bash
pip install tensorflowjs
```

## 📦 Dependencies

Lihat `requirements.txt` untuk daftar lengkap.