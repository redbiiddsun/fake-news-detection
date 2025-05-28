FROM python:3.12-slim

WORKDIR /app

COPY app/ /app/

COPY models/ /models/

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 3001

CMD ["python", "app.py"]