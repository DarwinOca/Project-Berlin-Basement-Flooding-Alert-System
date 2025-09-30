FROM python:3.11-slim

WORKDIR /app

COPY requirements_fe.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501
EXPOSE 8502

CMD ["streamlit", "run", "app.py"]