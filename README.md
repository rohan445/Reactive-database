# Reactive-database

Overview

This project is an AI-powered SQL query optimizer that predicts the execution time of queries and provides optimization suggestions. It utilizes a PostgreSQL database for storing executed queries and their execution times and uses TensorFlow to train a model for prediction.

Features
Logs SQL queries and their execution times into a PostgreSQL database.
Trains a machine learning model using TensorFlow to predict execution time.
Suggests query optimizations based on predictions.
Continuously learns and improves as more queries are logged.

Configuration

Update the PostgreSQL database credentials in the script:

DB_HOST = "your_host"
DB_PORT = "your_port"
DB_NAME = "your_db"
DB_USER = "your_user"
DB_PASSWORD = "your_password"

Usage
Run the script to start logging SQL queries and training the model:
python optimizer.py
Execute queries, and they will be stored in the database with their execution times.
The model trains periodically, refining its predictions over time.
View optimization suggestions for improving query performance.

Future Enhancements
Implement automatic query rewriting for optimization.
Improve model accuracy with additional query features.

