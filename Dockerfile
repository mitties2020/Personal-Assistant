FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

# 👇 IMPORTANT: module `app.py`, Flask instance `app`
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app:app"]
