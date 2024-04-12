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

## ACLARACION
Como se pidio tanto README.MD como un archivo .jpynb decidi hacer el README.MD en español y el challenge.jpynb en ingles. 

## IMPORTANTE
El archivo de datos Farmers-protest-tweets-2021-2-4.json esta incluido en la carpeta data/raw pero estaba comprimido, no tenía sentido subirlo sin comprimir donde pesaba más de 10 veces más. Es importante descomprimir este archivo en esa misma carpeta antes de ejecutar el código.

# Ejercicio 1 --> q1_memory.py

## Enfoque de la opcion

Utilice en esta opcion un codigo mas conciso y un poco mas legible, tratando de mantener la menor candidad de codigo y estructuras, no utilizo ```persist()``` para utillizar menos memoria. Pero al persistir datos hay un aumento en el tiempo de procesamiento, especialmente si se accede a los mismos datos varias veces.

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
Para ejecutar este script, necesitarás tener Python y Spark instalados en tu sistema. Aquí dejo una guía paso a paso:

### Instalar dependencias:
Asegúrate de tener Python y Apache Spark instalados.
Instala dependecias, si aún no están instalados:
Copy code
```bash 
pip install requirements.txt
```

### Preparar el archivo de datos:
El archivo ```farmers-protest-tweets-2021-2-4.json``` de datos esta en la carpeta data/raw, comprimido en un .rar es necesario descomprimirlo antes de ejecutar el codigo.

### Ejecutar el script:
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

# Ejercicio 2 --> q1_time.py

## Enfoque de la opcion

Utiliza persist(StorageLevel.MEMORY_AND_DISK) ```persist()``` para mantener los datos en memoria y en disco si es necesario, lo cual acelera el acceso a los datos repetidos.
Defino explícitamente un esquema de datos (CUSTOM_SCHEMA), para mejorar la eficiencia del procesamiento al evitar inferencia de tipos
Esta opcion es mas rapida, pero consume mas memoria que la anterior y el codigo es un poco mas largo y puede dificultar el mantenimiento.

## Descripción
El script realiza las siguientes operaciones:

1. Inicializa una sesión de Spark y lee los datos del archivo especificado.
2. Persiste el DataFrame en memoria y disco para optimizar el rendimiento.
3. Agrupa los datos por fecha de creación y cuenta los posts para cada fecha.
4. Filtra las 10 fechas con más publicaciones y obtiene los usuarios más activos en esas fechas.
5. Utilizamos una función de ventana para ordenar los usuarios por su actividad y selecciona el más activo para cada fecha.

## Cómo ejecutar el script
Para ejecutar este script, necesitarás tener Python y Spark instalados en tu sistema. Aquí dejo una guía paso a paso:

### Instalar dependencias:
Asegúrate de tener Python y Apache Spark instalados.
Instala dependecias, si aún no están instalados:
Copy code
```bash 
pip install requirements.txt
```
### Preparar el archivo de datos:
El archivo ```farmers-protest-tweets-2021-2-4.json``` de datos esta en la carpeta data/raw, comprimido en un .rar es necesario descomprimirlo antes de ejecutar el codigo. (Este paso es comun para todos los ejercicios pero lo vamos a comentar en cada caso, por si se quiere solo ejecutar un caso aislado.)

### Ejecutar el script:
Ejecuta el script desde la línea de comandos, pasando la ruta del archivo JSON como argumento:
bash
Copy code
```bash 
python q1_time.py /data/raw/farmers-protest-tweets-2021-2-4.json
```
### Revisar los resultados:
El script imprimirá los resultados en la consola, mostrando las 10 fechas con más publicaciones y obtiene los usuarios más activos en esas fechas.

# Ejercicio 3 --> q2_memory.py

## Enfoque de la opcion

El enfoque es similar en cada ejercicio entre eficiencia de tiempos y memoria.
En este enfoque, empleo un código más compacto y fácil de entender, procurando reducir al mínimo la cantidad de código y estructuras. Evito el uso de la función ```persist()``` para conservar memoria. Sin embargo, al persistir los datos, se observa un incremento en el tiempo de procesamiento.

## Descripción
El script realiza las siguientes operaciones:

1. Inicializa una sesión de Spark y lee datos de tweets del archivo JSON.
2. Extrae emojis del contenido del tweet utilizando una función definida por el usuario (UDF).
3. Cuenta las apariciones de cada emoji.
4. Ordena los emojis por frecuencia, con orden alfabético para desempates.
5. Devuelve los 10 emojis más utilizados junto con sus recuentos.

## Cómo ejecutar el script
Para ejecutar este script, necesitarás tener Python y Spark instalados en tu sistema. Aquí dejo una guía paso a paso:

### Instalar dependencias:
Asegúrate de tener Python y Apache Spark instalados.
Instala dependecias, si aún no están instalados:
Copy code
```bash 
pip install requirements.txt
```

### Preparar el archivo de datos:
El archivo ```farmers-protest-tweets-2021-2-4.json``` de datos esta en la carpeta data/raw, comprimido en un .rar es necesario descomprimirlo antes de ejecutar el codigo. (Este paso es comun para todos los ejercicios pero lo vamos a comentar en cada caso, por si se quiere solo ejecutar un caso aislado.)

### Ejecutar el script:
Ejecuta el script desde la línea de comandos, pasando la ruta del archivo JSON como argumento:
bash
Copy code
```bash 
python q2_memory.py /data/raw/farmers-protest-tweets-2021-2-4.json
```
### Revisar los resultados:
El script imprimirá los resultados en la consola, mostrando 10 emojis más utilizados junto con sus recuentos.

### Notas adicionales
Los resultados del análisis de uso de memoria se mostrarán en la consola si memory_profiler está activo.

# Ejercicio 4 --> q2_time.py

## Enfoque de la opcion

El enfoque es similar en cada ejercicio entre eficiencia de tiempos y memoria.
Utiliza la función ```persist()``` con el parámetro StorageLevel.MEMORY_AND_DISK para conservar los datos tanto en la memoria como en el disco si es necesario. Esto optimiza el acceso a los datos que se utilizan con frecuencia.
Establezco un esquema de datos personalizado (CUSTOM_SCHEMA) de manera explícita para mejorar la eficacia del procesamiento al evitar la necesidad de inferir tipos de datos.
Aunque esta opción ofrece mayor velocidad, requiere una cantidad mayor de memoria que la alternativa anterior y el código resultante puede ser un poco más extenso, lo que podría complicar el mantenimiento.

## Descripción
El script realiza las siguientes operaciones:

1. Inicializa una sesión de Spark y lee datos de tweets del archivo JSON.
2. Extrae emojis del contenido del tweet utilizando una función definida por el usuario (UDF).
3. Cuenta las apariciones de cada emoji.
4. Ordena los emojis por frecuencia, con orden alfabético para desempates.
5. Devuelve los 10 emojis más utilizados junto con sus recuentos.

## Cómo ejecutar el script
Para ejecutar este script, necesitarás tener Python y Spark instalados en tu sistema. Aquí dejo una guía paso a paso:

### Instalar dependencias:
Asegúrate de tener Python y Apache Spark instalados.
Instala dependecias, si aún no están instalados:
Copy code
```bash 
pip install requirements.txt
```

### Preparar el archivo de datos:
El archivo ```farmers-protest-tweets-2021-2-4.json``` de datos esta en la carpeta data/raw, comprimido en un .rar es necesario descomprimirlo antes de ejecutar el codigo. (Este paso es comun para todos los ejercicios pero lo vamos a comentar en cada caso, por si se quiere solo ejecutar un caso aislado.)

### Ejecutar el script:
Ejecuta el script desde la línea de comandos, pasando la ruta del archivo JSON como argumento:
bash
Copy code
```bash 
python q2_time.py /data/raw/farmers-protest-tweets-2021-2-4.json
```
### Revisar los resultados:
El script imprimirá los resultados en la consola, mostrando 10 emojis más utilizados junto con sus recuentos.

# Ejercicio 5 --> q3_memory.py

## Enfoque de la opcion

El enfoque es similar en cada ejercicio entre eficiencia de tiempos y memoria.
En este enfoque, empleo un código más compacto y fácil de entender, procurando reducir al mínimo la cantidad de código y estructuras. Evito el uso de la función ```persist()``` para conservar memoria. Sin embargo, al persistir los datos, se observa un incremento en el tiempo de procesamiento.

## Descripción
El script realiza las siguientes operaciones:

1. Inicializa una sesión de Spark y lee datos de tweets del archivo JSON.
2. Se utilizan expresiones regulares para extraer menciones del texto de las publicaciones.
3. Se eliminan las menciones vacías.
4. Se agrupan las menciones por publicación y se cuentan las ocurrencias.
5. Se ordenan las menciones por su frecuencia y se seleccionan las 10 más frecuentes.

## Cómo ejecutar el script
Para ejecutar este script, necesitarás tener Python y Spark instalados en tu sistema. Aquí dejo una guía paso a paso:

### Instalar dependencias:
Asegúrate de tener Python y Apache Spark instalados.
Instala dependecias, si aún no están instalados:
Copy code
```bash 
pip install requirements.txt
```

### Preparar el archivo de datos:
El archivo ```farmers-protest-tweets-2021-2-4.json``` de datos esta en la carpeta data/raw, comprimido en un .rar es necesario descomprimirlo antes de ejecutar el codigo. (Este paso es comun para todos los ejercicios pero lo vamos a comentar en cada caso, por si se quiere solo ejecutar un caso aislado.)

### Ejecutar el script:
Ejecuta el script desde la línea de comandos, pasando la ruta del archivo JSON como argumento:
bash
Copy code
```bash 
python q3_memory.py /data/raw/farmers-protest-tweets-2021-2-4.json
```
### Revisar los resultados:
El script imprimirá los resultados en la consola, mostrando las menciones por su frecuencia y se seleccionan las 10 más frecuentes.

### Notas adicionales
Los resultados del análisis de uso de memoria se mostrarán en la consola si memory_profiler está activo.