FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# data/ is mounted as a volume at runtime — this just ensures the dir exists
# in the image for runs without a volume mount
RUN mkdir -p data

EXPOSE 8000

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
