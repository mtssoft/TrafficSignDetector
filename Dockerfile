# Python 3.12 imajını temel alıyoruz
FROM python:3.12

# Çalışma dizinini belirliyoruz
WORKDIR /app

# Yerel bilgisayarınızdaki requirements.txt dosyasını konteynerin /app dizinine kopyalıyoruz
COPY requirements.txt /app/requirements.txt

# requirements.txt dosyasındaki kütüphaneleri yüklüyoruz
RUN pip install -r /app/requirements.txt

# Diğer dosyaları da kopyalayın (kod, veri, vb.)
COPY . /app

# Konteyner çalıştırıldığında yapılacak komut (örneğin, uygulamanızı başlatabilirsiniz)
CMD ["python", "src/trafficSignDetection.py"]
