# Veri işleme ve model oluşturma için gerekli kütüphaneler
# Dosya ve dizin işlemleri için
import os
# Etiketleri işlemek için
import pandas as pd
# Matematiksel işlemler ve diziler için
import numpy as np
# Derin öğrenme modeli oluşturmak için
import tensorflow as tf
# Katmanları sırayla oluşturmak için
from tensorflow.keras.models import Sequential
# CNN katmanları için
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# Veri setini eğitim/test olarak bölmek için
from sklearn.model_selection import train_test_split
# Etiketleri one-hot encoding'e çevirmek için
from tensorflow.keras.utils import to_categorical
# Etiketleri sayısal hale getirmek için
from sklearn.preprocessing import LabelEncoder
# görselleri yüklemek ve yeniden boyutlandırmak için kullanılır
from tensorflow.keras.preprocessing import image
# Convolution ve pooling katmanından sonra görüntünün halini kaydetmek için kullanılacak
from tensorflow.keras.models import Model
# Görselleştirme yapmak için
import matplotlib.pyplot as plt
# Eğitim, test veri setini bölmek için
from sklearn.model_selection import train_test_split

# 1. Veri yollarını tanımlanır.
# Görsellerin bulunduğu ana klasörün yolu.
data_dir = "../data/Trafik"
# Görsellerin etiketlerinin bulunduğu CSV dosyasının yolu.
labels_path = "../data/labels.csv"

# Görsel etiketlerinin bulunduğu csv dosyası yüklenir
labels_df = pd.read_csv(labels_path)
# Etiketleri görmek için ilk 5 dosya yazdırılır.
print(labels_df.head())
# Görselleri ve etiketleri eşleştirmek için diziler tanımlanır
image_paths = []
image_labels = []

# Her classid için döngü kurulur
for class_id in labels_df['Classid'].unique():
    # class_id'ye ait tüm görsellerin bulunduğu dizin okunur
    class_folder = os.path.join(data_dir, str(class_id)) 
    # Görseller listelenir
    for img_file in os.listdir(class_folder):
        if img_file.endswith('.png') or img_file.endswith('.jpg'):  # Görsellerin uzantısı
            img_path = os.path.join(class_folder, img_file)
            image_paths.append(img_path)
            image_labels.append(class_id)

# Etiketleri birleştirilir ve kategorik hale getirilir
labels = to_categorical(image_labels, num_classes=len(labels_df))

# CSV'den açıklama kolonunu almak için label_names sözlüğü oluşturulur
label_names = labels_df.set_index('Classid')['Name'].to_dict()

# Görselleri yüklemeek için kullanılacak metod (32x32 piksel boyutunda)
def load_and_preprocess_images(image_paths):
    images = []
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=(32, 32))  # Görseli 32x32 boyutunda yükleyin
        img_array = image.img_to_array(img)  # Görseli array'e çevir
        images.append(img_array)
    return np.array(images)

# Görseller yüklenir
images = load_and_preprocess_images(image_paths)
# Görsellerin boyutlarını kontrol edilir. Görsellerin (num_samples, 32, 32, 3) şeklinde olması gerek
print(images.shape)
# İlk olarak veri setini eğitim ve geçici (test + validation) olarak ayrılır
X_train, X_temp, y_train, y_temp = train_test_split(
    images, labels, test_size=0.3, random_state=42
)
# Ardından, geçici seti test ve doğrulama olarak ikiye bölünür
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
# Ayırma boyutlarını kontrol edilir
print("Eğitim seti boyutu:", X_train.shape)
print("Doğrulama seti boyutu:", X_val.shape)
print("Test seti boyutu:", X_test.shape)


# Örnek görselleri gösterme metodu
def show_example_images(image_paths, labels, label_names, num_examples=5):
    random_indices = np.random.choice(len(image_paths), num_examples, replace=False)
    for idx in random_indices:
        img = image.load_img(image_paths[idx], target_size=(32, 32))
        label_index = labels[idx].argmax()  # Etiketin numerik değeri
        label_text = label_names.get(label_index, f"Bilinmeyen Etiket ({label_index})")
        plt.figure()
        plt.imshow(img)
        plt.title(f'Etiket: {label_text}')
        plt.axis('off')
        plt.show()

# Veri setinden örnek görseller gösterilir
show_example_images(image_paths, labels, label_names, num_examples=5)


# Eğitim ve doğrulama kayıplarını çizdirip kaydetme metodu
# Eğitim ve doğrulama kayıplarını/grafikleri çizme
def show_graphic(history, metric):
    plt.figure()
    plt.plot(history.history[metric], label=f'Eğitim {metric}')
    plt.plot(history.history[f'val_{metric}'], label=f'Doğrulama {metric}')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.show()


# CNN modelini tasarımı tanımları

# Modelimiz sıralı olarak tasaralnıyor
model = Sequential()

# İlk Convolution katmanı ve MaxPooling
# 3x3 lük bir filtre uygulanacak Convulution katmanı tanımı. 32 filtre uygulanacaktır. Filtre sonrası ReLu aktivasyon fonksiyonu 
# uygulanarak negatif değer  olan siyahlar 0 yapılacak. input_shape ise girdi değerleridir. 
# 32x32 piksellik 3 renk kanalına sahip görüntü gireceğini belirtir.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), name="conv1"))
# Havuzlama katmanı değerleri. Burada 2x2'lik alanda max-pooling yapılacağı belirtilir. Yani conv katmanı ve relu sonrası elde edilen
# özellik matrisindeki her 2x2'lik alandaki maksimum değer alınarak matris boyutu küçültülecektir.
model.add(MaxPooling2D((2, 2), name="pool1"))

# İkinci Convolution katmanı ve MaxPooling
# 64 adet filtresi olan, 3x3'lük filtre uygulanan 2. convolution katmanı tanımı. Convolution sonrası tekrar ReLu uygulanır
# siyahlar 0'a çekilir.
model.add(Conv2D(64, (3, 3), activation='relu', name="conv2"))
# 2. havuzlama katmanı değerleri. Burada 2x2'lik alanda max-pooling yapılacağı belirtilir. Yani conv2 katmanı ve relu sonrası 
# elde edilen özellik matrisindeki her 2x2'lik alandaki maksimum değer alınarak matris boyutu küçültülecektir.
model.add(MaxPooling2D((2, 2), name="pool2"))

# Düzleştirme (Flatten) ve Dense katmanları
# Bu katmanda görüntünün sayısal matris temsili vektöre dönüştürülür. Fully connected katman (yoğun katman) vektör girdisi bekler
model.add(Flatten(name="flatten"))
# Fully connected katman modelin öğrenme işlemini yaptığı katmandır. 18 nörondan oluşur ve aktivasyon fonksiyonu relu'dur.
model.add(Dense(128, activation='relu', name="dense1"))
# Aşırı öğrenmeyi engellemek için tanımlanmış katmandır. 0.5 değeri her bir öğrenme döneminde nöronların %50'sinin devre dışı 
# bırakılacağını ifade eder.
model.add(Dropout(0.5, name="dropout"))
# Modelin tahminin üretildiği çıktı katmanıdır. aktivasyon fonksiyonu olarak relu kullanılmıştır.
model.add(Dense(len(labels_df), activation='softmax', name="output"))

# Modeli derleme
# Optimizasyon algoritması olarak adam, kayıp fonksiyonu olarak da çok sınıflı yapılandırmalarda yaygın kullanılan 
# categorical crossentropy kullanılmıştır.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Modelin özetini yazdırılır
model.summary()

# model eğitilir
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
     # Eğitim için 10 epoch belirlenmiştir
    epochs=10, 
    batch_size=32,
    # Eğitim süreci çıktısı yazdırılır
    verbose=1 
)

# Eğitim ve doğrulama doğruluk ve kayıp grafikleri çizilir
show_graphic(history, 'loss')
show_graphic(history, 'accuracy')

# Test setinde modelin doğruluğu ölçülür
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test doğruluğu: {test_acc * 100:.2f}%")

# Test veri setinden rastgele 5 görsel ve tahminlerin gösterilmesi
def show_predictions_on_test_set(model, X_test, y_test, label_names, num_examples=5):
    random_indices = np.random.choice(len(X_test), num_examples, replace=False)
    for idx in random_indices:
        img = X_test[idx]
        true_label_index = y_test[idx].argmax()  # Doğru etiketin numerik değeri
        true_label_text = label_names.get(true_label_index, f"Bilinmeyen Etiket ({true_label_index})")

        # Model tahmini
        pred_label_index = model.predict(img[np.newaxis, ...]).argmax()
        pred_label_text = label_names.get(pred_label_index, f"Bilinmeyen Tahmin ({pred_label_index})")

        # Görseli ve tahmini gösterme
        plt.figure()
        plt.imshow(img.astype('uint8'))
        plt.title(f'Etiket: {true_label_text}, Tahmin: {pred_label_text}')
        plt.axis('off')
        plt.show()

# Test veri setinde tahminler
show_predictions_on_test_set(model, X_test, y_test, label_names, num_examples=5)

# Modeli kaydetme
model.save('../models/traffic_sign_model.h5')

