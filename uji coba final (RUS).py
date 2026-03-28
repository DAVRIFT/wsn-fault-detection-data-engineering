# Mengimpor Library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Preprocessing Data

# 1.1 Data Cleaning
def clean_data(df):
    # Hapus baris dengan nilai yang hilang
    df_cleaned = df.dropna()
    # Hapus duplikat
    df_cleaned = df_cleaned.drop_duplicates()
    return df_cleaned

# 1.2 Normalisasi Data
def normalize_data(df):
    scaler = StandardScaler()
    columns_to_normalize = ['Humidity', 'Temperature']
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df, scaler

# 1.3 Label Encoding
def encode_labels(labels):
    le = LabelEncoder()
    return le.fit_transform(labels), le

# 2. Identifikasi Kesalahan melalui Simulasi
def simulate_faults(df):
    np.random.seed(42)
    df['Label'] = 'Normal'
    
    # Simulate Spike
    spike_idx = np.random.choice(df.index, size=int(0.1*len(df)), replace=False)
    df.loc[spike_idx, 'Humidity'] += np.random.uniform(2, 4, size=len(spike_idx))
    df.loc[spike_idx, 'Label'] = 'Spike'
    
    # Simulate Drift
    drift_idx = np.random.choice(df.index[~df.index.isin(spike_idx)], size=int(0.1*len(df)), replace=False)
    df.loc[drift_idx, 'Temperature'] += np.linspace(0, 2, len(drift_idx))
    df.loc[drift_idx, 'Label'] = 'Drift'
    
    # Simulate Bias
    bias_idx = np.random.choice(df.index[~df.index.isin(spike_idx) & ~df.index.isin(drift_idx)], size=int(0.1*len(df)), replace=False)
    df.loc[bias_idx, 'Humidity'] += 1.5
    df.loc[bias_idx, 'Label'] = 'Bias'
    
    return df

# Load data (assuming we have a CSV file named 'wsn_data.csv')
data = pd.read_csv('C:/Users/david/.spyder-py3/dataset_final.csv')

# Apply preprocessing steps
data_cleaned = clean_data(data)
data_normalized, scaler = normalize_data(data_cleaned)
data_with_faults = simulate_faults(data_normalized)

# Prepare features and labels
X = data_with_faults[['Humidity', 'Temperature']]
y = data_with_faults['Label']

# Encode labels
y_encoded, label_encoder = encode_labels(y)

# 3. Random Undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y_encoded)

# 4. Pembagian Dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.4, random_state=42)

# 5. Algoritma Klasifikasi Extra-Tree
etc = ExtraTreesClassifier(n_estimators=100, random_state=42)
etc.fit(X_train, y_train)

# 6. Deteksi Normal, Spike, Drift, Bias
y_pred = etc.predict(X_test)

# 7. Evaluasi Model
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance
feature_importance = etc.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
importance_df = importance_df.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Visualisasi Data
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_resampled['Humidity'], X_resampled['Temperature'], c=y_resampled, cmap='viridis')

plt.colorbar(scatter)
plt.title('Visualisasi Data WSN')
plt.xlabel('Humidity (Normalized)')
plt.ylabel('Temperature (Normalized)')
plt.show()

print("Proses Deteksi Kesalahan Data WSN Selesai.")

# Visualisasi Bentuk Gelombang untuk Setiap Jenis Kesalahan
def plot_waveforms(df):
    # Pisahkan data berdasarkan jenis kesalahan
    fault_types = ['Normal', 'Spike', 'Drift', 'Bias']
    
    plt.figure(figsize=(15, 10))
    
    for i, fault in enumerate(fault_types, 1):
        plt.subplot(2, 2, i)
        subset = df[df['Label'] == fault]
        plt.plot(subset.index, subset['Humidity'], label='Humidity', color='blue')
        plt.plot(subset.index, subset['Temperature'], label='Temperature', color='red')
        plt.title(f'{fault} Fault')
        plt.xlabel('Reading #')
        plt.ylabel('Sensor Value')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# Panggil fungsi untuk memvisualisasikan bentuk gelombang untuk setiap jenis kesalahan
plot_waveforms(data_with_faults)

# Membuat DataFrame untuk perbandingan prediksi dan aktual
comparison_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

# Decode angka kembali ke label
comparison_df['Actual_Label'] = label_encoder.inverse_transform(comparison_df['Actual'])
comparison_df['Predicted_Label'] = label_encoder.inverse_transform(comparison_df['Predicted'])

# Menambahkan kolom untuk menandai prediksi benar/salah
comparison_df['Is_Correct'] = comparison_df['Actual'] == comparison_df['Predicted']

# Menampilkan hasil perbandingan
print("\nPerbandingan Prediksi dan Data Aktual:")
print(comparison_df)

# Ringkasan hasil prediksi
print("\nRingkasan Prediksi:")
print(f"Total data uji: {len(comparison_df)}")
print(f"Prediksi benar: {sum(comparison_df['Is_Correct'])}")
print(f"Prediksi salah: {len(comparison_df) - sum(comparison_df['Is_Correct'])}")

# Menampilkan kesalahan prediksi
print("\nDetail Kesalahan Prediksi:")
incorrect_predictions = comparison_df[~comparison_df['Is_Correct']]
print(incorrect_predictions)

# Visualisasi dengan grafik gelombang
plt.figure(figsize=(15, 10))

# Plot untuk setiap jenis kesalahan
fault_types = ['Normal', 'Spike', 'Drift', 'Bias']
colors = ['blue', 'red', 'green', 'purple']

for fault, color in zip(fault_types, colors):
    # Data aktual
    fault_data = data_with_faults[data_with_faults['Label'] == fault]
    
    plt.subplot(2, 1, 1)
    plt.plot(fault_data.index[:100], fault_data['Humidity'][:100], 
             label=f'{fault} - Humidity', color=color, alpha=0.7)
    plt.title('Humidity Waveform by Fault Type')
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Humidity')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(fault_data.index[:100], fault_data['Temperature'][:100], 
             label=f'{fault} - Temperature', color=color, alpha=0.7)
    plt.title('Temperature Waveform by Fault Type')
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Temperature')
    plt.legend()

plt.tight_layout()
plt.show()

# Visualisasi distribusi prediksi vs aktual
plt.figure(figsize=(12, 6))

# Plot distribusi aktual
plt.subplot(1, 2, 1)
sns.countplot(data=comparison_df, x='Actual_Label')
plt.title('Distribusi Data Aktual')
plt.xticks(rotation=45)

# Plot distribusi prediksi
plt.subplot(1, 2, 2)
sns.countplot(data=comparison_df, x='Predicted_Label')
plt.title('Distribusi Data Prediksi')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Get actual and predicted labels in original format (first 100 samples)
y_test_original = label_encoder.inverse_transform(y_test)[:100]
y_pred_original = label_encoder.inverse_transform(y_pred)[:100]

# Create DataFrame for actual vs predicted values
comparison_df = pd.DataFrame({
    'Actual': y_test_original,
    'Predicted': y_pred_original
})

# Print actual vs predicted values
print("\nActual vs Predicted Values (First 100 samples):")
print(comparison_df)

# Calculate accuracy for first 100 samples
class_accuracy = {}
for class_name in label_encoder.classes_:
    mask = (comparison_df['Actual'] == class_name)
    class_total = mask.sum()
    correct_predictions = ((comparison_df['Actual'] == class_name) & 
                         (comparison_df['Predicted'] == class_name)).sum()
    accuracy = (correct_predictions / class_total) * 100 if class_total > 0 else 0
    class_accuracy[class_name] = accuracy

# Visualize actual vs predicted values over time
plt.figure(figsize=(15, 8))

# Create categorical colors for each class
unique_labels = label_encoder.classes_
color_map = dict(zip(unique_labels, sns.color_palette("husl", len(unique_labels))))

# Plot actual values
plt.plot(range(len(y_test_original)), 
         pd.Categorical(y_test_original).codes, 
         'b-', label='Actual', linewidth=2, alpha=0.7)

# Plot predicted values
plt.plot(range(len(y_pred_original)), 
         pd.Categorical(y_pred_original).codes, 
         'r--', label='Predicted', linewidth=2, alpha=0.7)

# Customize the plot
plt.title('Actual vs Predicted Values (First 100 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.yticks(range(len(unique_labels)), unique_labels)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# Add accuracy information as text box
accuracy_text = "Class-wise Accuracy (First 100 samples):\n"
for class_name, accuracy in class_accuracy.items():
    accuracy_text += f"{class_name}: {accuracy:.2f}%\n"

plt.figtext(1.02, 0.5, accuracy_text, fontsize=10, ha='left', va='center')

plt.tight_layout()
plt.show()

# Create a detailed accuracy visualization for first 100 samples
plt.figure(figsize=(10, 6))
accuracy_df = pd.DataFrame(list(class_accuracy.items()), 
                          columns=['Class', 'Accuracy'])
sns.barplot(x='Class', y='Accuracy', data=accuracy_df)
plt.title('Accuracy by Class (First 100 Samples)')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)

# Add value labels on top of each bar
for i, v in enumerate(accuracy_df['Accuracy']):
    plt.text(i, v + 1, f'{v:.2f}%', ha='center')

plt.tight_layout()
plt.show()