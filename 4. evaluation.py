from sklearn.metrics import classification_report, confusion_matrix

def evaluate(y_test, y_pred, label_names):
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=label_names))
    
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
