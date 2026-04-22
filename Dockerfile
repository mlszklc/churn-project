# 1. Hafif bir Python temel goruntusu seciyoruz
FROM python:3.11-slim

# 2. Konteyner icinde calisma klasoru olusturuyoruz
WORKDIR /app

# 3. Sistem bagimliliklarini yukluyoruz
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Gerekli kutuphane listesini kopyalayip yukluyoruz
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Tum proje dosyalarini kopyaliyoruz
COPY . .

# 6. start.sh dosyasini calistirılabilir yapiyoruz
RUN chmod +x start.sh

# 7. FastAPI (8000) ve Streamlit (8501) portlarini disariya aciyoruz
EXPOSE 8000 8501

# 8. Streamlit saglik kontrolu
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 9. Her iki servisi birden baslatiyoruz
ENTRYPOINT ["./start.sh"]
