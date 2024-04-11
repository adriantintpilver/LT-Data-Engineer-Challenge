# LT Challenge Solution

For challenge description go to: https://github.com/rtakeshi/latam-challenge/blob/main/latam-challenge.md

## Objectives

1. The primary goal of this case solution is to explore memory usage and optimization within a distributed and scalable data processing environment like Spark.
2. Exploring scalable cloud solutions such as Cloud Storage, Cloud Build, and Cloud Run in GCP.
3. Implement an reproductible environment using Docker ready to execute Jupyter and PySpark
4. Implement an automated Cloud Build CI for my Docker image to GCP Artifact Registry.
5. Establishing and adhering to a Git flow to establish an efficient workflow for organizing features, builds, and testing tasks.
6. Implementing Test-Driven Development (TDD) to address the challenge questions.
7. Implementing data transformations to define data quality layers.
8. Exploring analytical data transformation techniques to tackle the challenge questions.

As a bonus, I will also explore Infra as Code concepts to create a Google Cloud Storage (GCS) to store my test and staging datasets.

## Importantly
the data file farmers-protest-tweets-2021-2-4.json was included in the data/raw folder but was compressed, since it made no sense to upload it uncompressed where it weighed more than 10 times as much. It is important to unzip this file in that same folder before executing the code.

# ejercicio 1 --> q1_memory.py

## Descripción
El script realiza las siguientes operaciones:

1. Inicialización de Spark: Crea una sesión de Spark para procesar los datos.
2. Lectura del archivo JSON: Lee los datos de un archivo JSON especificado, usando un esquema definido que incluye campos como identificador, nombre de usuario, mensaje y fecha.
3. Análisis de datos:
    - Agrupa los datos por fecha y cuenta los mensajes para identificar las 10 fechas con más actividad.
    - Filtra los datos para quedarse solo con los mensajes de estas fechas.
    - Cuenta los mensajes de cada usuario en estas fechas para determinar el usuario más activo por día.
4. Resultados: El script devuelve una lista de tuplas, cada una conteniendo una fecha y el nombre del usuario más activo en esa fecha.

## Cómo ejecutar el script
Para ejecutar este script, necesitarás tener Python y Spark instalados en tu sistema. Aquí te dejo una guía paso a paso:

### Instalar dependencias:
Asegúrate de tener Python y Apache Spark instalados.
Instala pyspark y memory_profiler usando pip, si aún no están instalados:
Copy code
´´pip install pyspark memory_profiler´´

### Preparar el archivo de datos:
El archivo ´´farmers-protest-tweets-2021-2-4.json´´ de datos esta en la carpeta data/raw, comprimido en un .rar es necesario descomprimirlo antes de ejecutar el codigo.

### Ejecutar el script:
Guarda el código en un archivo, por ejemplo, analyze_tweets.py.
Ejecuta el script desde la línea de comandos, pasando la ruta del archivo JSON como argumento:
bash
Copy code
´´python q1_memory.py /data/raw/farmers-protest-tweets-2021-2-4.json´´

### Revisar los resultados:
El script imprimirá los resultados en la consola, mostrando las fechas con más mensajes y el usuario más activo en esas fechas.

### Notas adicionales
Los resultados del análisis de uso de memoria se mostrarán en la consola si memory_profiler está activo.

