"""
Sentiment Classifier
Trains a simple machine learning model to predict whether text is
positive or negative, then lets you test it on your own sentences.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------
# Step 1: Training data
# In a real project this would come from a large dataset file.
# Kept small and readable here so you can see exactly what the
# model is learning from.
# ---------------------------------------------------------

texts = [
    "I love this product, it works perfectly",
    "Amazing experience, would buy again",
    "This is the best purchase I've made",
    "Fantastic quality and fast delivery",
    "Really happy with how this turned out",
    "Excellent service, very satisfied",
    "This made my day so much better",
    "Highly recommend this to everyone",
    "I like this a lot, it's great",
    "This is so much fun to use",
    "What a wonderful surprise this was",
    "I'm really enjoying this so far",
    "This is exactly what I wanted",
    "Such a great experience overall",
    "I'm impressed with how well this works",
    "This made me smile all day",
    "Perfect, couldn't ask for more",
    "I'm so glad I tried this",
    "This exceeded my expectations completely",
    "Absolutely delightful from start to finish",
    "Terrible experience, would not recommend",
    "This is the worst product I've ever bought",
    "Completely disappointed with the quality",
    "Waste of money, do not buy this",
    "Awful service, never coming back",
    "This broke after one day of use",
    "Very frustrating and poorly made",
    "I regret purchasing this item",
    "I hate how this turned out",
    "This is so boring and pointless",
    "I really dislike this a lot",
    "This was a huge letdown",
    "Nothing about this worked properly",
    "I'm so annoyed with this whole thing",
    "This is a complete disaster",
    "I can't stand using this anymore",
    "This ruined my entire day",
    "Such a disappointing and sad outcome",
    "I wish I had never bought this",
    "This is painfully bad and slow",
]

labels = [
    "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive",
    "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative",
]


# ---------------------------------------------------------
# Step 2: Convert text into numbers the model can learn from
# TF-IDF scores each word by how important it is to a sentence,
# relative to how common it is across all sentences.
# ---------------------------------------------------------

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)


# ---------------------------------------------------------
# Step 3: Evaluate using cross-validation
# With a dataset this small, a single train/test split is unreliable
# (a handful of unlucky test examples can make accuracy swing wildly).
# Cross-validation instead trains and tests the model multiple times,
# each time on a different slice of the data, then averages the results
# for a much more stable estimate.
# ---------------------------------------------------------

model = MultinomialNB()
scores = cross_val_score(model, X, labels, cv=5)

print(f"Cross-validation accuracy per fold: {[f'{s:.0%}' for s in scores]}")
print(f"Average accuracy: {scores.mean():.0%}\n")


# ---------------------------------------------------------
# Step 4: Train the final model on all available data
# (cross-validation above was just for evaluation; now we train
# on everything so predictions below use all the examples we have)
# ---------------------------------------------------------

model.fit(X, labels)


# ---------------------------------------------------------
# Step 5: Try it on your own sentences
# ---------------------------------------------------------

def predict_sentiment(sentence):
    vector = vectorizer.transform([sentence])
    return model.predict(vector)[0]


def main():
    print("Type a sentence to check its sentiment (or 'quit' to exit)\n")
    while True:
        sentence = input("Your sentence: ")
        if sentence.lower() == "quit":
            break
        result = predict_sentiment(sentence)
        print(f"Predicted sentiment: {result}\n")


if __name__ == "__main__":
    main()