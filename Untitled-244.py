"""
==================================================
PROGRAM KLASIFIKASI DRY BEAN DENGAN KNN
==================================================
"""

# 1. IMPORT LIBRARY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import warnings, os, urllib.request, zipfile
warnings.filterwarnings('ignore')

print("="*55)
print("PROGRAM KLASIFIKASI DRY BEAN DENGAN KNN")
print("="*55)

# 2. LOAD DATASET
os.makedirs('data', exist_ok=True)
file_path = 'data/DryBeanDataset.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("✅ Dataset loaded from local!")
else:
    print("📥 Downloading dataset...")
    urllib.request.urlretrieve("https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip", 'data/dry_bean.zip')
    with zipfile.ZipFile('data/dry_bean.zip', 'r') as zip_ref:
        zip_ref.extractall('data/')
    for file in os.listdir('data/'):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join('data/', file))
            df.to_csv(file_path, index=False)
            break
    print("✅ Dataset downloaded!")

print(f"📊 Data: {df.shape[0]} rows, {df.shape[1]} columns")

# 3. IDENTIFIKASI TARGET
target_col = [col for col in df.columns if col.lower() == 'class']
target_col = target_col[0] if target_col else df.columns[-1]
feature_cols = [col for col in df.columns if col != target_col]

print(f"🎯 Target: '{target_col}' ({df[target_col].nunique()} classes)")

# 4. CEK KESEIMBANGAN KELAS
class_counts = df[target_col].value_counts()
print("\n📊 Class distribution:")
for cls, cnt in class_counts.items():
    print(f"   {str(cls)[:12]:12s}: {cnt:5d} ({cnt/len(df)*100:5.1f}%)")

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(x=[str(c)[:10] for c in class_counts.index], y=class_counts.values, ax=ax[0], palette='Set2')
ax[0].set_title('Class Distribution')
ax[0].tick_params(axis='x', rotation=45)
ax[1].pie(class_counts.values, labels=[str(c)[:10] for c in class_counts.index], autopct='%1.1f%%')
ax[1].set_title('Class Percentage')
plt.tight_layout()
plt.savefig('1_class_distribution.png')
plt.show()

# 5. PREPROCESSING
le = LabelEncoder()
y = le.fit_transform(df[target_col].astype(str))
X = df[feature_cols].select_dtypes(include=[np.number])
class_names = [str(c) for c in le.classes_]

# SMOTE jika tidak seimbang
ratio = class_counts.min() / class_counts.max()
if ratio < 0.5:
    print(f"\n⚠️ Data imbalance (ratio={ratio:.3f}), applying SMOTE...")
    X, y = SMOTE(random_state=42).fit_resample(X, y)
    print(f"✅ After SMOTE: {len(X)} samples")

# Normalisasi
X = StandardScaler().fit_transform(X)
print("✅ Data normalized")

# 6. PAIRPLOT
df_plot = pd.DataFrame(X[:, :5], columns=feature_cols[:5])
df_plot['Class'] = [class_names[i] for i in y]
sns.pairplot(df_plot, hue='Class', diag_kind='hist', plot_kws={'alpha': 0.5, 's': 5})
plt.suptitle('Pairplot of Top 5 Features', y=1.02)
plt.savefig('2_pairplot.png', dpi=150, bbox_inches='tight')
plt.show()

# 7. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n📚 Training: {len(X_train)} samples, 📖 Testing: {len(X_test)} samples")

# 8. CARI K OPTIMAL
grid = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': list(range(1, 31))}, cv=5)
grid.fit(X_train, y_train)
k_opt = grid.best_params_['n_neighbors']
print(f"✅ Optimal K = {k_opt} (Accuracy: {grid.best_score_:.4f})")

# Plot K optimization
plt.figure(figsize=(10, 5))
plt.plot(range(1, 31), grid.cv_results_['mean_test_score'], 'b-o')
plt.axvline(k_opt, color='r', linestyle='--', label=f'K = {k_opt}')
plt.xlabel('K Value'), plt.ylabel('Accuracy'), plt.title('K Value Optimization')
plt.legend(), plt.grid(True, alpha=0.3)
plt.savefig('3_k_optimal.png')
plt.show()

# 9. TRAINING & EVALUASI
knn = KNeighborsClassifier(n_neighbors=k_opt)
knn.fit(X_train, y_train)

# Training metrics
y_pred_train = knn.predict(X_train)
print(f"\n📊 TRAINING: Accuracy={accuracy_score(y_train, y_pred_train):.4f}, F1={f1_score(y_train, y_pred_train, average='weighted'):.4f}")

# Confusion Matrix Training
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_train, y_pred_train), annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix - Training (K={k_opt})')
plt.xlabel('Predicted'), plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('4_cm_training.png')
plt.show()

# Testing metrics
y_pred_test = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, average='weighted')
rec = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')

print(f"\n📊 TESTING RESULTS:")
print(f"   Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"   Precision: {prec:.4f} ({prec*100:.2f}%)")
print(f"   Recall   : {rec:.4f} ({rec*100:.2f}%)")
print(f"   F1 Score : {f1:.4f}")

# Confusion Matrix Testing
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt='d', cmap='Greens', 
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix - Testing (K={k_opt})')
plt.xlabel('Predicted'), plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('5_cm_testing.png')
plt.show()

# Classification Report
print(f"\n📋 CLASSIFICATION REPORT (Testing):")
print(classification_report(y_test, y_pred_test, target_names=class_names))

# 10. PERFORMANCE COMPARISON
comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Training': [accuracy_score(y_train, y_pred_train), 
                 precision_score(y_train, y_pred_train, average='weighted'),
                 recall_score(y_train, y_pred_train, average='weighted'),
                 f1_score(y_train, y_pred_train, average='weighted')],
    'Testing': [acc, prec, rec, f1]
})

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(comparison['Metric']))
ax.bar(x - 0.175, comparison['Training'], 0.35, label='Training', color='#3498db')
ax.bar(x + 0.175, comparison['Testing'], 0.35, label='Testing', color='#e74c3c')
ax.set_xticks(x), ax.set_xticklabels(comparison['Metric'])
ax.set_ylabel('Score'), ax.set_title('Training vs Testing Performance')
ax.legend(), ax.set_ylim(0, 1.1)
for i, (t, ts) in enumerate(zip(comparison['Training'], comparison['Testing'])):
    ax.text(i - 0.175, t + 0.02, f'{t:.3f}', ha='center', fontsize=9)
    ax.text(i + 0.175, ts + 0.02, f'{ts:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('6_performance_comparison.png')
plt.show()

# 11. SIMPAN HASIL
with open('RESULTS.txt', 'w', encoding='utf-8') as f:
    f.write("="*50 + "\n")
    f.write("HASIL KLASIFIKASI DRY BEAN DATASET\n")
    f.write("="*50 + "\n\n")
    f.write(f"Optimal K Value     : {k_opt}\n")
    f.write(f"Training Accuracy   : {accuracy_score(y_train, y_pred_train)*100:.2f}%\n")
    f.write(f"Testing Accuracy    : {acc*100:.2f}%\n")
    f.write(f"Testing Precision   : {prec*100:.2f}%\n")
    f.write(f"Testing Recall      : {rec*100:.2f}%\n")
    f.write(f"Testing F1 Score    : {f1*100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred_test, target_names=class_names))

print("\n✅ Results saved:")
print("   📁 RESULTS.txt")
print("   📁 1_class_distribution.png")
print("   📁 2_pairplot.png")
print("   📁 3_k_optimal.png")
print("   📁 4_cm_training.png")
print("   📁 5_cm_testing.png")
print("   📁 6_performance_comparison.png")

print("\n" + "="*55)
print("🎉 PROGRAM SELESAI!")
print("="*55)