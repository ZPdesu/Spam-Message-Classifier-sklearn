from pymongo import MongoClient
import datetime
client = MongoClient()
db = client.test_database

post = {"author": "Mike",
        "text": "My first blog post!",
        "tags": ["mongodb", "python", "pymongo"]}
posts = db['posts']
posts.insert_one(post)
print 'ok'
