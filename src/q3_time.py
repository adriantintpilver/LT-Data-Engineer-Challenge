from typing import List, Tuple
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DateType
from pyspark.sql.functions import regexp_extract, collect_list, explode, count
from pyspark.storagelevel import StorageLevel

CUSTOM_SCHEMA = StructType([
    StructField("identifier", StringType(), True),
    StructField("user_name", StringType(), True),
    StructField("post_text", StringType(), True),
    StructField("creation_date", DateType(), True) 
])

def q3_time(file_path: str) -> List[Tuple[str, int]]:

    spark = SparkSession.builder.appName("FarmersProtestTweetsOptimization").getOrCreate()

    df = spark.read.option('delimiter', '~').option('header', True).option('multiline', True).schema(CUSTOM_SCHEMA).csv(file_path)
    df.persist(StorageLevel.MEMORY_AND_DISK)

    # Mention regex pattern
    mention_pattern = r"(@\w+)"

    # Extracting all strings with mention pattern
    mentions_df = df.select("post_text", regexp_extract(df["post_text"], mention_pattern, 1).alias("mention"))

    # Removing all empty mentions
    mentions_df = mentions_df.filter(mentions_df["mention"] != "")

    # Collecting all mentions as an array
    mentions_df = mentions_df.groupBy("post_text").agg(collect_list("mention").alias("mentions"))

    # DataFrame containing all user mentions
    user_mentions = mentions_df.select("post_text", explode(mentions_df["mentions"]).alias("mention"))

    # Count and group by user mentions
    mention_counts = user_mentions.groupBy("mention").agg(count("mention").alias("count"))

    # Ordering by mentions count and mention username for tie edge case
    sorted_mentions = mention_counts.orderBy(mention_counts["count"].desc(), mention_counts["mention"]).limit(10)

    result_collection = sorted_mentions.collect()

    # Creating result list of tuples
    result = []
    for row in result_collection:
        result.append((row['mention'], row['count']))

    return result
