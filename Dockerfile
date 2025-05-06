FROM python:3.10-slim

# Instala dependencias
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copia tu script
COPY script.py .

# Define variables de entorno por defecto
ENV inputDataset=/input
ENV outputDir=/output

# Permite sobreescribir parámetros en tiempo de ejecución
ENTRYPOINT ["python3", "script.py"]
CMD ["--input-dir", "/input", "--output-dir", "/output"]
