#!/bin/bash

# FastAPI'yi arka planda baslat
uvicorn app:app --host 0.0.0.0 --port 8000 &

# FastAPI hazir olana kadar bekle (max 30 saniye)
echo "FastAPI bekleniyor..."
for i in $(seq 1 30); do
    curl -s http://127.0.0.1:8000/health > /dev/null && break
    sleep 1
done
echo "FastAPI hazir, Streamlit baslatiliyor..."

# Streamlit'i on planda baslat
streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0
