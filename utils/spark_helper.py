from pyspark.sql import SparkSession


def create_spark_session(app_name: str = "DataChefETL") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()
