import xgboost
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = xgboost.XGBClassifier()
model.fit(X_train, y_train)

# Evaluate
print(model.score(X_test, y_test))
