from typing import List, Tuple
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, row_number
from pyspark.sql.window import Window
from pyspark.sql.types import DateType, StringType, StructField, StructType
from pyspark.storagelevel import StorageLevel
from memory_profiler import profile


CUSTOM_SCHEMA = StructType([
    StructField("identifier", StringType(), True),
    StructField("user_name", StringType(), True),
    StructField("post_text", StringType(), True),
    StructField("creation_date", DateType(), True) 
])

@profile
def custom_time_query(file_path: str) -> List[Tuple[datetime.date, str]]:
        
    spark_session = SparkSession.builder.appName("CustomTweetsOptimization").getOrCreate()


    data_frame = spark_session.read.option('delimiter', '~').option('header', True).option('multiline', True).schema(CUSTOM_SCHEMA).csv(file_path)
    data_frame.persist(StorageLevel.MEMORY_AND_DISK)

    #Top 10 dates with more content
    date_counts = data_frame.groupBy('creation_date').agg(count('post_text').alias('date_count'))
    date_counts = date_counts.orderBy(col('date_count').desc()).limit(10)

    #Joined DF filtering only top 10 dates by inner joining
    filtered_data_frame = data_frame.join(date_counts, 'creation_date', 'inner')

    #Counting Users posts on top 10 dates
    user_counts_by_date = filtered_data_frame.groupBy('creation_date', 'date_count', 'user_name').agg(count('post_text').alias('user_count'))

    #Creating a window analytical function to filter top 1 user_name in each date
    #Edge case: if there is a tie, the user_name will follow alphabetical ordering
    window_specification = Window.partitionBy('creation_date', 'date_count').orderBy(col('user_count').desc(), col('user_name'))

    #creating rank based on row_number ordering
    user_counts_by_date = user_counts_by_date.withColumn('rank', row_number().over(window_specification))

    #getting the Rank1 user_name for each date
    top_user_names_by_date = user_counts_by_date.filter(user_counts_by_date['rank'] == 1)

    #ordering by content count by date
    top_user_names_by_date = top_user_names_by_date.select(['creation_date', 'user_name']).orderBy(col('date_count').desc())

    # Collect dataframe results
    result_collection = top_user_names_by_date.collect()

    #Creating result list of tuples
    result = []
    for row in result_collection:
        result.append((row['creation_date'], row['user_name']))

    return result
