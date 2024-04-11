# LT Challenge, Solucion

Para obtener una descripción del desafío, visite: https://github.com/rtakeshi/latam-challenge/blob/main/latam-challenge.md

## Objetivos

1. El objetivo principal de la solución de este caso es explorar el uso y la optimización de la memoria dentro de un entorno de procesamiento de datos distribuido y escalable como Spark.
2. Explorar soluciones de nube escalables como Cloud Storage, Cloud Build y Cloud Run en GCP.
3. Implementar un entorno reproducible usando Docker listo para ejecutar Jupyter y PySpark
4. Implementar un CI de Cloud Build automatizado para mi imagen de Docker en GCP Artifact Registry.
5. Establecer y adherirse a un flujo de Git para establecer un flujo de trabajo eficiente para organizar funciones, compilaciones y tareas de prueba.
6. Implementar el desarrollo basado en pruebas (TDD) para abordar las preguntas desafiantes.
7. Implementar transformaciones de datos para definir capas de calidad de datos.
8. Explorar técnicas analíticas de transformación de datos para abordar las preguntas desafiantes.

## IMPORTANTE
El archivo de datos Farmers-protest-tweets-2021-2-4.json esta incluido en la carpeta data/raw pero estaba comprimido, no tenía sentido subirlo sin comprimir donde pesaba más de 10 veces más. Es importante descomprimir este archivo en esa misma carpeta antes de ejecutar el código.

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
```bash 
pip install pyspark memory_profiler
```

### Preparar el archivo de datos:
El archivo ```farmers-protest-tweets-2021-2-4.json``` de datos esta en la carpeta data/raw, comprimido en un .rar es necesario descomprimirlo antes de ejecutar el codigo.

### Ejecutar el script:
Guarda el código en un archivo, por ejemplo, analyze_tweets.py.
Ejecuta el script desde la línea de comandos, pasando la ruta del archivo JSON como argumento:
bash
Copy code
```bash 
python q1_memory.py /data/raw/farmers-protest-tweets-2021-2-4.json
```

### Revisar los resultados:
El script imprimirá los resultados en la consola, mostrando las fechas con más mensajes y el usuario más activo en esas fechas.

### Notas adicionales
Los resultados del análisis de uso de memoria se mostrarán en la consola si memory_profiler está activo.