import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.tokenize import word_tokenize

# Read dataset
data = pd.read_csv('data\\data.csv', sep=';')

# Pre-process dataset (remove punctuation, convert to lowercase, tokenize text)
nltk.download('punkt')
data['Message'] = (data['Message'].apply(lambda x: ' '.join(word_tokenize(str(x).lower()))))

# Split into train, test
X = data['Message']
y = data['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a BoW representation
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Create and train Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_bow, y_train)

# Create and train decision tree
decision_tree_classifier = DecisionTreeClassifier(random_state=42)
decision_tree_classifier.fit(X_train_bow, y_train)

# Make predictions (to verify models)
naive_bayes_predictions = naive_bayes_classifier.predict(X_test_bow)
decision_tree_predictions = decision_tree_classifier.predict(X_test_bow)

# Evaluate models performance
print("Naive Bayes Accuracy:", accuracy_score(y_test, naive_bayes_predictions))
print("Naive Bayes Classification Report:")
print(classification_report(y_test, naive_bayes_predictions))

print("Decision Tree Accuracy:", accuracy_score(y_test, decision_tree_predictions))
print("Decision Tree Classification Report:")
print(classification_report(y_test, decision_tree_predictions))

# Confusion Matrix plot
naive_bayes_confusion = confusion_matrix(y_test, naive_bayes_predictions)
seaborn.heatmap(naive_bayes_confusion, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Naive Bayes)")
plt.show()

decision_tree_confusion = confusion_matrix(y_test, decision_tree_predictions)
seaborn.heatmap(decision_tree_confusion, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Decision Tree)")
plt.show()
