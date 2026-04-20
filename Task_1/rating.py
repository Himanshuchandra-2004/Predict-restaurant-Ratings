import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# loading the dataset
df = pd.read_csv('Dataset_internship.csv')
print("Dataset shape:", df.shape)

# these columns either leak info about the target or are just IDs/addresses
# so we drop them before training
cols_to_drop = ['Restaurant ID', 'Restaurant Name', 'Address',
                'Locality', 'Locality Verbose',
                'Rating color', 'Rating text']
df.drop(columns=cols_to_drop, inplace=True)

# checking for missing values
print("\nMissing values:")
missing = df.isnull().sum()
print(missing[missing > 0])

# only Cuisines has nulls, filling with 'Unknown'
df['Cuisines'] = df['Cuisines'].fillna('Unknown')
print("Filled missing cuisines with 'Unknown'")

# encoding the binary yes/no columns to 1/0
yes_no_cols = ['Has Table booking', 'Has Online delivery',
               'Is delivering now', 'Switch to order menu']
for col in yes_no_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# for city, cuisines and currency we use label encoding
encoder = LabelEncoder()
for col in ['City', 'Cuisines', 'Currency']:
    df[col] = encoder.fit_transform(df[col].astype(str))

# separating features and target
X = df.drop(columns=['Aggregate rating'])
y = df['Aggregate rating']

print(f"\nUsing {len(X.columns)} features: {list(X.columns)}")
print(f"Target range: {y.min()} to {y.max()}")

# splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# defining the models we want to compare
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=8, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10,
                                           random_state=42, n_jobs=-1)
}

# training each model and storing results
results = {}
print("\n" + "=" * 50)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # calculating metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # accuracy = how many predictions are within 0.5 of the actual rating
    tolerance = 0.5
    within_tolerance = np.sum(np.abs(y_test.values - y_pred) <= tolerance)
    accuracy = (within_tolerance / len(y_test)) * 100

    # r2 as a percentage (how much variance the model explains)
    r2_accuracy = max(r2, 0) * 100

    results[name] = {
        'model': model,
        'predictions': y_pred,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Accuracy (±0.5)': accuracy,
        'R2 Accuracy %': r2_accuracy
    }

    print(f"\n{name}")
    print("-" * 40)
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R2   : {r2:.4f}")
    print(f"  Accuracy (±0.5 tolerance): {accuracy:.2f}%")
    print(f"  R2 Accuracy: {r2_accuracy:.2f}%")

# let's look at which features matter most according to random forest
print("\n" + "=" * 50)
print("Top 10 important features (Random Forest):\n")

rf = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
print(importances.head(10).to_string())

# showing a few sample predictions to see how the best model is doing
print("\n" + "=" * 50)
print("Sample predictions from Random Forest:\n")

actual = y_test.values[:5]
predicted = results['Random Forest']['predictions'][:5]

for i in range(len(actual)):
    err = abs(actual[i] - predicted[i])
    print(f"  Row {i+1} -> Actual: {actual[i]:.1f}, Predicted: {predicted[i]:.2f}, Error: {err:.2f}")

print("\nDone!")

# ── VISUALIZATIONS ──────────────────────────────────────────────
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Distribution of ratings
plt.figure()
sns.histplot(y, bins=20, kde=True, color='steelblue')
plt.title('Distribution of Aggregate Ratings')
plt.xlabel('Aggregate Rating')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plot_rating_distribution.png', dpi=150)
plt.show()
print("Saved: plot_rating_distribution.png")

# 2. Correlation heatmap
plt.figure(figsize=(12, 8))
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, square=True)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('plot_correlation_heatmap.png', dpi=150)
plt.show()
print("Saved: plot_correlation_heatmap.png")

# 3. Feature importance (Random Forest) bar chart
plt.figure()
top_features = importances.head(10)
sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('plot_feature_importance.png', dpi=150)
plt.show()
print("Saved: plot_feature_importance.png")

# 4. Actual vs Predicted scatter plot (Random Forest)
plt.figure()
rf_preds = results['Random Forest']['predictions']
sns.scatterplot(x=y_test, y=rf_preds, alpha=0.4, color='teal', edgecolor=None)
plt.plot([0, 5], [0, 5], 'r--', label='Perfect prediction')  # diagonal line
plt.title('Actual vs Predicted Ratings (Random Forest)')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.legend()
plt.tight_layout()
plt.savefig('plot_actual_vs_predicted.png', dpi=150)
plt.show()
print("Saved: plot_actual_vs_predicted.png")

# 5. Model comparison - R2 and Accuracy side by side
model_names = list(results.keys())
r2_scores = [results[m]['R2 Accuracy %'] for m in model_names]
acc_scores = [results[m]['Accuracy (±0.5)'] for m in model_names]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(x=model_names, y=r2_scores, palette='Blues_d', ax=axes[0])
axes[0].set_title('R² Accuracy (%)')
axes[0].set_ylabel('Percentage')
axes[0].set_ylim(0, 100)
for i, v in enumerate(r2_scores):
    axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

sns.barplot(x=model_names, y=acc_scores, palette='Greens_d', ax=axes[1])
axes[1].set_title('Accuracy within ±0.5 Rating (%)')
axes[1].set_ylabel('Percentage')
axes[1].set_ylim(0, 100)
for i, v in enumerate(acc_scores):
    axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

plt.suptitle('Model Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_model_comparison.png', dpi=150)
plt.show()
print("Saved: plot_model_comparison.png")

print("\nAll plots saved!")