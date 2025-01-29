from pymongo import MongoClient

# MongoDB connection URI
client = MongoClient("mongodb://localhost:27017/")

# Specify the database name
db = client["practice1"]  # Replace "practice" with your desired database name
