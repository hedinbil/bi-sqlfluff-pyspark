result = (
    spark.sql("SELECT * FROM orders WHERE date >= '2024-01-01'")
    .filter("status = 'completed'")
    .select("order_id", "customer_id", "total")
    .orderBy("total", ascending=False)
    .limit(100)
    .cache()
)
print(f"Processed {result.count()} orders")
