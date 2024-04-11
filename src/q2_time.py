from typing import List, Tuple
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, explode, count
from pyspark.sql.types import DateType, StringType, StructField, StructType, ArrayType
from pyspark.storagelevel import StorageLevel
import emoji

# Custom schema for staging data
CUSTOM_SCHEMA = StructType([
    StructField("identifier", StringType(), True),
    StructField("user_name", StringType(), True),
    StructField("post_text", StringType(), True),
    StructField("creation_date", DateType(), True) 
])

def q2_time(file_path: str) -> List[Tuple[str, int]]:

    # Initialize Spark session
    spark = SparkSession.builder.appName("FarmersProtestTweetsOptimization").getOrCreate()

    # Read data from file using custom schema and persist in memory
    df = spark.read.option('delimiter', '~').option('header', True).option('multiline', True).schema(CUSTOM_SCHEMA).csv(file_path)
    df.persist(StorageLevel.MEMORY_AND_DISK)

    # Define a function to extract emojis from text
    def extract_emojis(text):
        if text is not None:
            return emoji.get_emoji_regexp().findall(text)
        else:
            return []
      
    # Define UDF for extracting emojis
    extract_emojis_udf = udf(extract_emojis, ArrayType(StringType()))

    # Create a new DataFrame with exploded array of content to find emojis
    emoji_df = df.withColumn('emoji', explode(extract_emojis_udf(df['post_text'])))

    # Count and order by usage and emoji alphabetical order for tie edge case
    emoji_counts = emoji_df.groupBy('emoji').agg(count('emoji').alias('count')).orderBy(col("count").desc(), col('emoji')).limit(10)

    # Collect the result DataFrame into a list of tuples
    result_collection = emoji_counts.collect()

    # Create result list of tuples
    result = []
    for row in result_collection:
        result.append((row['emoji'], row['count']))

    return result
