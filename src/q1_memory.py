from typing import List, Tuple
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, row_number
from pyspark.sql.window import Window
from pyspark.sql.types import DateType, StringType, StructField, StructType
from memory_profiler import profile

# Data structure definition
TEMP_SCHEMA = StructType([
    StructField("identifier", StringType(), True),
    StructField("user_name", StringType(), True),
    StructField("message", StringType(), True),
    StructField("date", DateType(), True)
])

@profile
def analyze_memory_usage(file_path: str) -> List[Tuple[datetime.date, str]]:
    
    # Initialize Spark
    spark_session = SparkSession.builder.appName("TweetsAnalysis").getOrCreate()
    
    # Reading the file
    data = spark_session.read.option('delimiter', '~').option('header', True).option('multiline', True).schema(TEMP_SCHEMA).csv(file_path)
    
    # The 10 dates with the most messages
    date_count = data.groupBy('date').agg(count('message').alias('date_count'))
    date_count = date_count.orderBy(col('date_count').desc()).limit(10)

    # Joining data filtering only the top 10 dates
    filtered_data = data.join(date_count, 'date', 'inner')

    # Counting user posts on the top 10 dates
    user_count_by_date = filtered_data.groupBy('date', 'date_count', 'user_name').agg(count('message').alias('user_count'))

    # Window analytic function to filter the top user_name by date
    window_spec = Window.partitionBy('date', 'date_count').orderBy(col('user_count').desc(), col('user_name'))

    # Creating the ranking
    user_count_by_date = user_count_by_date.withColumn('rank', row_number().over(window_spec))

    # Getting the top user_name for each date
    top_users_by_date = user_count_by_date.filter(user_count_by_date['rank'] == 1)

    # Sorting by message count by date
    top_users_by_date = top_users_by_date.select(['date', 'user_name']).orderBy(col('date_count').desc())

    # Collecting the results
    results_collection = top_users_by_date.collect()

    # Creating the results list
    results = []
    for row in results_collection:
        results.append((row['date'], row['user_name']))

    return results

