from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("test").getOrCreate()

# First query
df1 = spark.sql("SELECT * FROM table1 WHERE id = 1")

# Second query with method call
df2 = spark.sql("SELECT name FROM users").filter("active = 1")

# Process results
result = df1.join(df2, "id")
result.show()
