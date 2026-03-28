from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

def split_data(X, y):
    return train_test_split(X, y, test_size=0.4, random_state=42)

def train_model(X_train, y_train):
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
