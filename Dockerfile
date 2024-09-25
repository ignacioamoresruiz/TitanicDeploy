# Usar la imagen oficial de Python
FROM python:3.12

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de requerimientos y luego instalar las dependencias
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . .

# Exponer el puerto 8080 para Google Cloud Run
EXPOSE 8080

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
