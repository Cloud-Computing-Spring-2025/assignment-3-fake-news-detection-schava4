from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

# Initialize Spark session
spark_session = SparkSession.builder.appName("FakeNewsDetection").getOrCreate()

# ---------------------------
# Task 1: Load & basic exploration
# ---------------------------
# Load fake news CSV into DataFrame
news_df = spark_session.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

# Create temporary view for SQL queries
news_df.createOrReplaceTempView("news_data")

# Run basic exploratory queries
news_df.show(5)
print("Total number of news articles:", news_df.count())
news_df.select("label").distinct().show()

# Save first 5-row sample to CSV
news_df.limit(5).toPandas().to_csv("task1_output.csv", index=False)

# ---------------------------
# Task 2: Text preprocessing
# ---------------------------
# Normalize text column to lowercase
normalized_df = news_df.withColumn("text", lower(col("text")))

# Tokenize text into words
tokenizer = Tokenizer(inputCol="text", outputCol="words")
tokenized_df = tokenizer.transform(normalized_df)

# Remove stop words from tokenized words
stopword_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
cleaned_df = stopword_remover.transform(tokenized_df)

# Select columns for Task 2 output
task2_df = cleaned_df.select("id", "title", "filtered_words", "label")

# Save Task 2 output to CSV
task2_df.toPandas().to_csv("task2_output.csv", index=False)

# ---------------------------
# Task 3: Feature extraction
# ---------------------------
# Apply HashingTF to generate raw term frequencies
hashing_tf = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)
hashed_df = hashing_tf.transform(task2_df)

# Compute inverse document frequency (IDF)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(hashed_df)
featurized_df = idf_model.transform(hashed_df)

# Index string labels to numeric indices
label_indexer = StringIndexer(inputCol="label", outputCol="label_index")
final_df = label_indexer.fit(featurized_df).transform(featurized_df)

# Select columns for Task 3 output
task3_df = final_df.select("id", "filtered_words", "features", "label_index")

# Save Task 3 output to CSV
task3_df.toPandas().to_csv("task3_output.csv", index=False)

# ---------------------------
# Task 4: Model training
# ---------------------------
# Prepare title DataFrame for joining with predictions
title_df = cleaned_df.select("id", "title")

# Split into training and test sets
train_df, test_df = task3_df.randomSplit([0.8, 0.2], seed=42)

# Train logistic regression model
logistic_regression = LogisticRegression(featuresCol="features", labelCol="label_index")
lr_model = logistic_regression.fit(train_df)

# Generate predictions on test set
predictions_df = lr_model.transform(test_df)

# Add original titles to prediction results
predictions_with_title_df = predictions_df.join(title_df, on="id", how="left")

# Select relevant columns and save Task 4 output
predictions_with_title_df \
    .select("id", "title", "label_index", "prediction") \
    .toPandas().to_csv("task4_output.csv", index=False)

# ---------------------------
# Task 5: Model evaluation
# ---------------------------
# Set up evaluators for accuracy and F1 score
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label_index", predictionCol="prediction", metricName="accuracy"
)
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label_index", predictionCol="prediction", metricName="f1"
)

# Compute metrics
accuracy_value = accuracy_evaluator.evaluate(predictions_df)
f1_value = f1_evaluator.evaluate(predictions_df)

# Save evaluation results to CSV
evaluation_results = pd.DataFrame({
    'Metric': ['Accuracy', 'F1 Score'],
    'Value': [accuracy_value, f1_value]
})
evaluation_results.to_csv("task5_output.csv", index=False)

# Print evaluation results
print("Model evaluation results:")
print(evaluation_results)

# ---------------------------
# Finish
# ---------------------------
spark_session.stop()