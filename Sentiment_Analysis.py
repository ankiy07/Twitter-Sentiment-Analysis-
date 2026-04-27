# !pip install pandas openpyxl


import numpy as np
import pandas as pd


twitter_data = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1")


# number of row and column
twitter_data.shape


twitter_data.head()


## giving name to coulmn and read data again

column_names=['target','id','date','flag','user','text']
twitter_data=twitter_data = pd.read_csv("training.1600000.processed.noemoticon.csv", names=column_names, encoding="ISO-8859-1")


twitter_data.shape


twitter_data.head()


# countinig number of missing value in our dataset
twitter_data.isnull().sum()


# checking the distribution
twitter_data['target'].value_counts()


twitter_data.replace({'target':{4:1}} , inplace=True)


# checking the distribution
twitter_data['target'].value_counts()


# it is a process of reducing a given word to its root word
# example: running , runs, ran all threats as run .


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


port_stem= PorterStemmer()


negation_words = {"not", "no", "nor", "never", "neither", "nobody", "nothing", "nowhere", "hardly", "barely", "scarcely"}
stop_words = set(stopwords.words('english')) - negation_words 

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)


twitter_data.head()


print(twitter_data['stemmed_content'])


# seperating the data and label
X = twitter_data['stemmed_content'].values
Y= twitter_data['target'].values


print(X)


print(Y)


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


print(X.shape,X_train.shape, X_test.shape)


print(X_train)


print(X_test)


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=500000,
    ngram_range=(1, 2),  
    min_df=2,
    max_df=0.95,
    sublinear_tf=True     
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


print(X_train)


print(X_test)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=2.0, solver='saga', max_iter=1000, n_jobs=-1)
model.fit(X_train, Y_train)


model.fit(X_train,Y_train)


# accuracy score on the training data
from sklearn.metrics import accuracy_score

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy score on the training data : ', training_data_accuracy) 


# accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy score on the test data : ', test_data_accuracy)


import pickle

filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))
pickle.dump(vectorizer, open('vectorizer.sav', 'wb'))


# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# testing on a random sample from the test data (e.g., index 200)
X_new = X_test[200]
print("Actual Label:", Y_test[200])

prediction = loaded_model.predict(X_new)
print("Predicted Label:", prediction)

if (prediction[0] == 0):
  print('Negative Tweet')
else:
  print('Positive Tweet')


from sklearn.metrics import confusion_matrix, classification_report, f1_score

# 1. Get the Confusion Matrix (TP, TN, FP, FN)
# Assuming 1 is Positive and 0 is Negative
tn, fp, fn, tp = confusion_matrix(Y_test, X_test_prediction).ravel()

# 2. Calculate Sensitivity and Specificity
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# 3. Get the F1 Score
f1 = f1_score(Y_test, X_test_prediction)

print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print("-" * 30)
print(f"Sensitivity (Recall): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"F1-Score: {f1:.2f}")

# 4. Detailed Report (includes R1/Recall and Precision)
print("\nFull Classification Report:")
print(classification_report(Y_test, X_test_prediction))


 


