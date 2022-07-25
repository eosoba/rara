
import pandas as pd
import numpy as np
import matplotlib as plt
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import confusion_matrix



# Get working directory, list files in working dir

os.getcwd()
os.listdir(os.getcwd())

# list files in organ directory

organ = os.listdir("fullerton/Organs_(anatomy)/")
rare = os.listdir("fullerton/Rare_diseases/")


#read a single text file
fname = organ[0]
print fname
f = open("fullerton/Organs_(anatomy)/" + fname,'r')
text = f.read()




# get all the text files in organ and rare
organ_text = [open("fullerton/Organs_(anatomy)/"+f).read() for f in organ if f.endswith('txt')]
rare_text = [open("fullerton/Rare_diseases/"+f).read() for f in rare if f.endswith('txt')]

# combine them together, make a seperate list for labels 
combined = organ_text + rare_text
labels =  ['organ' for i in range(len(organ_text))] + ['rare' for i in range(len(rare_text))]

# get count vectorizer
vectorizer = CountVectorizer()
print vectorizer


# look at stopwords
trial  = combined[0]
analyze = vectorizer.build_analyzer()
print analyze(trial)
vectorizer = CountVectorizer(stop_words = "english")
analyze = vectorizer.build_analyzer()
print analyze(trial)
print vectorizer.get_stop_words()



# Go over the size of feature matrix
vectorizer = CountVectorizer(max_features=10, stop_words="english")
tdmatrix = vectorizer.fit_transform(combined)
print tdmatrix
vectorizer.get_feature_names()


# get tfidf 
tfidf = TfidfTransformer()
yak = tfidf.fit_transform(tdmatrix)


#classification using td-idf
count_vect = CountVectorizer(max_features=100, decode_error='ignore', stop_words = "english")
tfidf = TfidfTransformer()

tdmatrix = count_vect.fit_transform(combined)
X = tfidf.fit_transform(tdmatrix)
y= labels
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = DecisionTreeClassifier(max_depth = 4)
#clf = LogisticRegression()
#clf = RandomForestClassifier()
clf.fit(X_train,y_train)

print "test set metrics: ", confusion_matrix(y_test, clf.predict(X_test))
print "training set metrics: ", confusion_matrix(y_train, clf.predict(X_train))

# get decision tree depth
clf.tree_.max_depth


# putting everything together
organ = os.listdir("fullerton/Organs_(anatomy)/")
rare = os.listdir("fullerton/Rare_diseases/")

organ_text = [open("fullerton/Organs_(anatomy)/"+f).read() for f in organ if f.endswith('txt')]
rare_text = [open("fullerton/Rare_diseases/"+f).read() for f in rare if f.endswith('txt')]

combined = organ_text + rare_text
labels =  ['organ' for i in range(len(organ_text))] + ['rare' for i in range(len(rare_text))]

count_vect = CountVectorizer(max_features=100, decode_error='ignore', stop_words = "english")
tfidf = TfidfTransformer()

tdmatrix = count_vect.fit_transform(combined)
X = tfidf.fit_transform(tdmatrix)
y= labels

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = DecisionTreeClassifier(max_depth = 4)
clf.fit(X_train,y_train)

print "test set metrics: ", confusion_matrix(y_test, clf.predict(X_test))
print "training set metrics: ", confusion_matrix(y_train, clf.predict(X_train))


# An aside. How I got the plot for the lecture. Get the 30 most common words in organ collection
from collections import Counter
giant = " ".join(organ_text)
giant = giant.lower()
giant = giant.replace("="," ",)
giant = giant.replace(")"," ")
giant = giant.replace( "(", " ")
giant = giant.replace(".", " ")
giant = giant = giant.split()
mscm = Counter(giant).most_common(30)
wd = np.array([w[1] for w in mscm])
what = pd.DataFrame(mscm)
what.index = what[0]
what[1].plot.bar()
plt.title("30 Most Common Words")
plt.savefig('mscm.png')

