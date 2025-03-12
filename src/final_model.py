import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading X and y from the HDF5 file
with h5py.File('dataset.h5', 'r') as f:
    X = np.array(f['X'])
    y = np.array(f['y'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Defining the model
model = LogisticRegression(max_iter=1000)

# Training
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Storing the model in an HDF5 file for deployment
with h5py.File('lr_model.h5', 'w') as f:
    f.create_dataset('model', data=model.coef_)
    f.create_dataset('intercept', data=model.intercept_)

print("Model trained on the dataset and saved in 'lr_model.h5'")
