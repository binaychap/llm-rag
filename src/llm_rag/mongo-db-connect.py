from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# Connection URI (update username/password if needed)
MONGO_URI = "mongodb://user:pass@127.0.0.1:27017/?directConnection=true"

def connect_to_mongo():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

        # Trigger connection
        client.admin.command("ping")
        print("✅ Connected to MongoDB!")

        return client

    except ConnectionFailure:
        print("❌ Failed to connect to MongoDB server")
    except OperationFailure as e:
        print(f"❌ Authentication failed: {e}")

if __name__ == "__main__":
    client = connect_to_mongo()

    if client:
        # Access database
        db = client["test_db"]

        # Access collection
        collection = db["test_collection"]

        # Insert sample document
        result = collection.insert_one({"name": "Binay", "role": "developer"})
        print(f"Inserted document ID: {result.inserted_id}")

        # Fetch document
        doc = collection.find_one({"name": "Binay"})
        print("Fetched document:", doc)