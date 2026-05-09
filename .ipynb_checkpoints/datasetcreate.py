import pandas as pd
import random

positive_feedback = [
    "This product is amazing and works perfectly",
    "Excellent quality and very useful",
    "I really like this product",
    "Great purchase and worth the money",
    "Very satisfied with this item",
    "Fantastic product and fast delivery",
    "Superb quality and easy to use",
    "Highly recommend this product",
    "Works exactly as expected",
    "Very happy with the performance"
]

negative_feedback = [
    "This product is terrible and stopped working",
    "Very poor quality and disappointing",
    "I hate this product",
    "Waste of money and useless",
    "Not satisfied with this item",
    "Bad product and slow delivery",
    "Poor build quality",
    "Completely disappointed with this purchase",
    "Does not work as expected",
    "Very bad experience with this product"
]

data = []

# generate 200 samples
for i in range(100):
    data.append({
        "feedback": random.choice(positive_feedback),
        "label": "positive"
    })
    
    data.append({
        "feedback": random.choice(negative_feedback),
        "label": "negative"
    })

df = pd.DataFrame(data)

df.to_csv("feedback.csv", index=False)

print("feedback.csv dataset created successfully!")