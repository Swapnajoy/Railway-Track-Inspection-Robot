import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Loading X and y from the HDF5 file
with h5py.File('dataset.h5', 'r') as f:
    X = f['X'][:] 
    y = f['y'][:]

# Defining models in a dictionary
models = {
    "LR": LogisticRegression(max_iter=1000),
    "LDA": LinearDiscriminantAnalysis(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "CART": DecisionTreeClassifier(),
    "RF": RandomForestClassifier(n_estimators=100),
    "NB": GaussianNB(),
    "SVM": SVC(kernel="linear")
}

# Performing 5-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

# Scaling the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Storing accuracy scores for each model
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    results[name] = scores  

# Converting results to a boxplot-friendly format
data = [results[name] for name in models]

# Creating Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=data)
plt.xticks(ticks=range(len(models)), labels=models.keys())
plt.ylabel("Accuracy")
plt.xlabel("ML Algorithms")
plt.title("Accuracy Comparison of ML Models with 5-Fold CV")
plt.grid(True)
plt.show()