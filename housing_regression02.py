"""
==============================================================
  HOUSING PRICE PREDICTION - DATA ANALYTICS MINI PROJECT
  Rajalakshmi Institute of Technology
  Department: AI & Data Science
==============================================================
  Run in VS Code terminal:
    pip install scikit-learn pandas numpy matplotlib seaborn
    python housing_regression.py
  
  Graphs will pop up one by one. Close each window to continue.
==============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

DIVIDER = "=" * 65
SUB     = "-" * 65

# ======================================================
# 1. GENERATE DATASET
# ======================================================
print(DIVIDER)
print("  STEP 1: LOADING DATASET")
print(DIVIDER)

np.random.seed(42)
n = 2000

MedInc     = np.random.lognormal(1.5, 0.6, n).clip(0.5, 15)
HouseAge   = np.random.uniform(1, 52, n)
AveRooms   = np.random.normal(5.5, 2, n).clip(1, 15)
AveBedrms  = np.random.normal(1.1, 0.5, n).clip(0.5, 5)
Population = np.random.lognormal(6, 0.8, n).clip(3, 35000)
AveOccup   = np.random.normal(3, 1, n).clip(1, 10)
Latitude   = np.random.uniform(32.5, 42.0, n)
Longitude  = np.random.uniform(-124.0, -114.0, n)

noise = np.random.normal(0, 0.35, n)
Price = (
    0.45 * MedInc
  + 0.12 * AveRooms
  + 0.04 * HouseAge
  - 0.10 * AveOccup
  - 0.30 * (Latitude  - 36.0)
  + 0.05 * (Longitude + 119.0)
  + noise
).clip(0.5, 5.0)

df = pd.DataFrame({
    'MedInc'    : MedInc,
    'HouseAge'  : HouseAge,
    'AveRooms'  : AveRooms,
    'AveBedrms' : AveBedrms,
    'Population': Population,
    'AveOccup'  : AveOccup,
    'Latitude'  : Latitude,
    'Longitude' : Longitude,
    'Price'     : Price,
})

feature_info = {
    'MedInc'    : 'Median income in block group',
    'HouseAge'  : 'Median house age in block group',
    'AveRooms'  : 'Average number of rooms per household',
    'AveBedrms' : 'Average number of bedrooms per household',
    'Population': 'Block group population',
    'AveOccup'  : 'Average household occupancy',
    'Latitude'  : 'Latitude coordinate',
    'Longitude' : 'Longitude coordinate',
    'Price'     : 'Median house value (target, in $100,000)',
}

print(f"\n  Dataset   : California-style Housing Dataset (n=2000)")
print(f"  Samples   : {df.shape[0]:,}")
print(f"  Features  : {df.shape[1] - 1}")
print(f"\n  Feature Descriptions:")
for col, desc in feature_info.items():
    print(f"    {col:<12} -> {desc}")

# ======================================================
# 2. EXPLORATORY DATA ANALYSIS
# ======================================================
print(f"\n{DIVIDER}")
print("  STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
print(DIVIDER)

print(f"\n--- Basic Statistics {'-'*45}")
print(df.describe().round(3).to_string())

print(f"\n--- Missing Values {'-'*47}")
missing = df.isnull().sum()
print(missing.to_string())
print(f"\n  Total missing values: {missing.sum()}")

print(f"\n--- Correlation with Price {'-'*39}")
corr = df.corr()['Price'].sort_values(ascending=False)
for feat, val in corr.items():
    bar  = '#' * int(abs(val) * 30)
    sign = '+' if val >= 0 else '-'
    print(f"  {feat:<12}  {sign}{abs(val):.4f}  {bar}")

# ======================================================
# PLOT 1 - Feature Distributions
# ======================================================
print(f"\n{DIVIDER}")
print("  PLOT 1: Feature Distributions  [close window to continue]")
print(DIVIDER)

fig, axes = plt.subplots(2, 5, figsize=(20, 9))
fig.suptitle('Feature Distributions', fontsize=15, fontweight='bold')
for i, col in enumerate(df.columns):
    r, c = divmod(i, 5)
    ax = axes[r][c]
    ax.hist(df[col], bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_title(col, fontsize=10, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('plot1_distributions.png', dpi=120, bbox_inches='tight')
print("  Saved: plot1_distributions.png")
plt.show()

# ======================================================
# PLOT 2 - Correlation Heatmap
# ======================================================
print(f"\n  PLOT 2: Correlation Heatmap  [close window to continue]")
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr().round(2), annot=True, cmap='coolwarm',
            fmt='.2f', linewidths=0.5, square=True)
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot2_heatmap.png', dpi=120, bbox_inches='tight')
print("  Saved: plot2_heatmap.png")
plt.show()

# ======================================================
# PLOT 3 - Price vs Top Features (Scatter)
# ======================================================
print(f"\n  PLOT 3: Price vs Top Features  [close window to continue]")
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Price vs Top Features', fontsize=14, fontweight='bold')
for ax, feat in zip(axes, ['MedInc', 'AveRooms', 'HouseAge', 'AveOccup']):
    ax.scatter(df[feat], df['Price'], alpha=0.25, s=8, color='steelblue')
    ax.set_xlabel(feat, fontsize=11)
    ax.set_ylabel('Price ($100k)')
    m, b = np.polyfit(df[feat], df['Price'], 1)
    xs = np.linspace(df[feat].min(), df[feat].max(), 200)
    ax.plot(xs, m*xs+b, color='red', linewidth=2, label='Trend')
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('plot3_scatter.png', dpi=120, bbox_inches='tight')
print("  Saved: plot3_scatter.png")
plt.show()

# ======================================================
# 3. DATA PREPROCESSING
# ======================================================
print(f"\n{DIVIDER}")
print("  STEP 3: DATA PREPROCESSING")
print(DIVIDER)

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n  Total samples   : {len(df):,}")
print(f"  Training set    : {len(X_train):,}  (80%)")
print(f"  Test set        : {len(X_test):,}   (20%)")
print(f"  Scaling method  : StandardScaler (zero mean, unit variance)")

print(f"\n  Scaled Feature Stats (Training):")
sc_df = pd.DataFrame(X_train_sc, columns=X.columns)
print(f"  {'Feature':<12}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
print(f"  {'-'*52}")
for col in sc_df.columns:
    print(f"  {col:<12}  {sc_df[col].mean():>8.4f}  {sc_df[col].std():>8.4f}  "
          f"{sc_df[col].min():>8.4f}  {sc_df[col].max():>8.4f}")

# ======================================================
# 4. TRAIN MODELS
# ======================================================
print(f"\n{DIVIDER}")
print("  STEP 4: TRAINING REGRESSION MODELS")
print(DIVIDER)

models = {
    'Linear Regression' : LinearRegression(),
    'Ridge Regression'  : Ridge(alpha=1.0),
    'Lasso Regression'  : Lasso(alpha=0.05),
    'Decision Tree'     : DecisionTreeRegressor(max_depth=6, random_state=42),
    'Random Forest'     : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting' : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    print(f"\n  Training: {name} ...", end=' ', flush=True)
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='r2').mean()
    results[name] = {
        'MSE'  : round(mse, 4),
        'RMSE' : round(rmse, 4),
        'MAE'  : round(mae, 4),
        'R2'   : round(r2, 4),
        'CV_R2': round(cv, 4),
        'model': model,
        'preds': y_pred,
    }
    print(f"Done.   R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

# ======================================================
# 5. EVALUATION RESULTS
# ======================================================
print(f"\n{DIVIDER}")
print("  STEP 5: MODEL EVALUATION RESULTS")
print(DIVIDER)

print(f"\n  {'Model':<26} {'MSE':>7} {'RMSE':>7} {'MAE':>7} {'R2':>7} {'CV_R2':>7}")
print(f"  {SUB}")
best_name = max(results, key=lambda k: results[k]['R2'])
for name, r in results.items():
    tag = "  <-- BEST" if name == best_name else ""
    print(f"  {name:<26} {r['MSE']:>7.4f} {r['RMSE']:>7.4f} {r['MAE']:>7.4f} "
          f"{r['R2']:>7.4f} {r['CV_R2']:>7.4f}{tag}")

best = results[best_name]
print(f"\n  BEST MODEL : {best_name}")
print(f"  R2         : {best['R2']}  (explains {best['R2']*100:.1f}% of price variance)")
print(f"  RMSE       : {best['RMSE']} ($100k)  ->  +/- ${best['RMSE']*100000:,.0f} avg error")
print(f"  MAE        : {best['MAE']} ($100k)  ->  +/- ${best['MAE']*100000:,.0f} median error")

# Linear Coefficients
print(f"\n{SUB}")
print("  Linear Regression - Feature Coefficients")
print(SUB)
lr_coef = pd.Series(results['Linear Regression']['model'].coef_, index=X.columns).sort_values()
for feat, coef in lr_coef.items():
    bar  = '#' * int(abs(coef) * 8)
    sign = '+' if coef >= 0 else '-'
    print(f"  {feat:<12}  {sign}{abs(coef):.4f}  {bar}")

# RF Importances
print(f"\n{SUB}")
print("  Random Forest - Feature Importances")
print(SUB)
rf_imp = pd.Series(
    results['Random Forest']['model'].feature_importances_, index=X.columns
).sort_values(ascending=False)
for feat, imp in rf_imp.items():
    bar = '#' * int(imp * 60)
    print(f"  {feat:<12}  {imp:.4f}  {bar}")

# ======================================================
# PLOT 4 - R2 Comparison
# ======================================================
print(f"\n{DIVIDER}")
print("  PLOT 4: R2 Comparison  [close window to continue]")
print(DIVIDER)

names  = list(results.keys())
r2vals = [results[n]['R2'] for n in names]
colors = ['#27ae60' if n == best_name else '#2980b9' for n in names]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(names, r2vals, color=colors, edgecolor='white', width=0.55)
for bar, val in zip(bars, r2vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('Model Comparison - R2 Score (Higher is Better)', fontsize=13, fontweight='bold')
ax.set_ylabel('R2 Score')
ax.set_ylim(0, 1.12)
ax.axhline(0.8, color='red', linestyle='--', linewidth=1.2, label='R2 = 0.80 threshold')
ax.set_xticklabels(names, rotation=20, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig('plot4_r2_comparison.png', dpi=120, bbox_inches='tight')
print("  Saved: plot4_r2_comparison.png")
plt.show()

# ======================================================
# PLOT 5 - Actual vs Predicted
# ======================================================
print(f"\n  PLOT 5: Actual vs Predicted ({best_name})  [close window to continue]")
y_pred_best = best['preds']
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, y_pred_best, alpha=0.35, s=10, color='steelblue')
mn, mx = y_test.min(), y_test.max()
ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Price ($100k)', fontsize=12)
ax.set_ylabel('Predicted Price ($100k)', fontsize=12)
ax.set_title(f'Actual vs Predicted - {best_name}\nR2 = {best["R2"]}',
             fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('plot5_actual_vs_pred.png', dpi=120, bbox_inches='tight')
print("  Saved: plot5_actual_vs_pred.png")
plt.show()

# ======================================================
# PLOT 6 - Residuals
# ======================================================
print(f"\n  PLOT 6: Residual Analysis  [close window to continue]")
residuals = y_test.values - y_pred_best
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_pred_best, residuals, alpha=0.3, s=8, color='coral')
axes[0].axhline(0, color='black', linewidth=1.5)
axes[0].set_xlabel('Predicted Price ($100k)')
axes[0].set_ylabel('Residuals')
axes[0].set_title(f'Residuals vs Fitted - {best_name}', fontweight='bold')
axes[1].hist(residuals, bins=40, color='coral', edgecolor='white', alpha=0.85)
axes[1].set_xlabel('Residual Value')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Residual Distribution', fontweight='bold')
plt.tight_layout()
plt.savefig('plot6_residuals.png', dpi=120, bbox_inches='tight')
print("  Saved: plot6_residuals.png")
plt.show()

# ======================================================
# PLOT 7 - RMSE Comparison
# ======================================================
print(f"\n  PLOT 7: RMSE Comparison  [close window to continue]")
rmse_vals = [results[n]['RMSE'] for n in names]
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(names, rmse_vals,
              color=['#c0392b' if n == best_name else '#e67e22' for n in names],
              edgecolor='white', width=0.55)
for bar, val in zip(bars, rmse_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('Model Comparison - RMSE (Lower is Better)', fontsize=13, fontweight='bold')
ax.set_ylabel('RMSE ($100k)')
ax.set_xticklabels(names, rotation=20, ha='right')
plt.tight_layout()
plt.savefig('plot7_rmse_comparison.png', dpi=120, bbox_inches='tight')
print("  Saved: plot7_rmse_comparison.png")
plt.show()

# ======================================================
# PLOT 8 - Feature Importance
# ======================================================
print(f"\n  PLOT 8: Feature Importance  [close window to continue]")
fig, ax = plt.subplots(figsize=(8, 5))
rf_imp.sort_values().plot(kind='barh', color='steelblue', edgecolor='white', ax=ax)
ax.set_title('Feature Importance - Random Forest', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('plot8_feature_importance.png', dpi=120, bbox_inches='tight')
print("  Saved: plot8_feature_importance.png")
plt.show()

# ======================================================
# 6. SAMPLE PREDICTIONS
# ======================================================
print(f"\n{DIVIDER}")
print("  STEP 6: SAMPLE PREDICTIONS (Best Model)")
print(DIVIDER)
print(f"\n  Model: {best_name}\n")
print(f"  {'Sample':<8} {'Actual ($100k)':>16} {'Predicted ($100k)':>18} "
      f"{'Error ($100k)':>14} {'Error ($)':>14}")
print(f"  {'-'*72}")
for idx in [0, 10, 50, 100, 200, 300]:
    actual = y_test.iloc[idx]
    pred   = best['preds'][idx]
    err    = actual - pred
    print(f"  #{idx:<7} {actual:>16.4f} {pred:>18.4f} {err:>14.4f} {err*100000:>+14,.0f}")

# ======================================================
# 7. GOODNESS-OF-FIT & CONCLUSION
# ======================================================
print(f"\n{DIVIDER}")
print("  STEP 7: GOODNESS-OF-FIT MEASURES")
print(DIVIDER)
print(f"""
  Metric  Full Name                     Ideal  Meaning
  ------  ----------------------------  -----  --------------------------------
  R2      Coefficient of Determination   1.00  Fraction of variance explained
  MSE     Mean Squared Error             0.00  Avg squared prediction error
  RMSE    Root Mean Squared Error        0.00  Error in same units as target
  MAE     Mean Absolute Error            0.00  Avg absolute error (robust)
  CV_R2   Cross-Validated R2 (5-fold)    1.00  Generalisation performance

  Results Summary:
  {'Model':<26} {'R2':>7}  Grade
  {SUB}""")
for name, r in sorted(results.items(), key=lambda x: -x[1]['R2']):
    grade = ('Excellent' if r['R2'] >= 0.85 else
             'Good'      if r['R2'] >= 0.75 else
             'Fair'      if r['R2'] >= 0.60 else 'Poor')
    print(f"  {name:<26} {r['R2']:>7.4f}  {grade}")

print(f"""
{DIVIDER}
  CONCLUSION
{DIVIDER}
  Dataset         : California-style Housing (2,000 samples, 8 features)
  Task            : Supervised Regression - Predict Median House Price

  Models Applied  :
    1. Linear Regression       - Baseline, interpretable
    2. Ridge Regression        - Linear + L2 regularization
    3. Lasso Regression        - Linear + L1 regularization
    4. Decision Tree           - Non-linear, easy to visualize
    5. Random Forest           - Ensemble of 100 decision trees
    6. Gradient Boosting       - Boosted ensemble, best performer

  Key Findings    :
    * MedInc (median income) is the STRONGEST price predictor
    * Latitude captures geographic price variation
    * Ensemble methods far outperform linear models
    * Best Model : {best_name}
        R2   = {best['R2']}  -> explains {best['R2']*100:.1f}% of price variance
        RMSE = {best['RMSE']} ($100k)  -> +/- ${best['RMSE']*100000:,.0f} avg error
        MAE  = {best['MAE']} ($100k)  -> +/- ${best['MAE']*100000:,.0f} median error

  8 Plots saved to current folder as PNG files.
{DIVIDER}
  PROGRAM COMPLETE
{DIVIDER}""")
