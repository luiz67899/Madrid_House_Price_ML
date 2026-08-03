
# %% [markdown]
# # 🏠 Madrid House Price Prediction
# ## A Data Science Approach
#
# **Author:** Luiz Augusto  
# **Dataset:** [Madrid Real Estate Market - Kaggle](https://www.kaggle.com/datasets/mirbektoktogaraev/madrid-real-estate-market)  
# **Objective:** Build a regression model to predict house prices (`buy_price`) in Madrid using structured features.
#
# ---
#
# ## Project Structure
# 1. Data Loading & Initial Exploration
# 2. Data Quality Assessment (Missing Values Analysis)
# 3. Feature Selection (Multiple Approaches)
# 4. Outlier Detection & Treatment
# 5. Train/Test Split (BEFORE any transformation — avoiding data leakage)
# 6. Preprocessing Pipeline
# 7. Model Training & Comparison
# 8. Error Analysis (Train vs Test — Bias/Variance Diagnosis)
# 9. Hyperparameter Tuning
# 10. Final Model Evaluation & Conclusions
#
# ---
#
# ### Why this structure?
# The order of operations matters in ML. A common mistake is to preprocess data (remove outliers,
# normalize, impute) **before** splitting into train/test. This causes **data leakage** — information
# from the test set influences preprocessing decisions, leading to optimistic (unrealistic) metrics.
#
# **Correct pipeline:** Load → Split → Preprocess (fit on train only) → Train → Evaluate

# %%
# === IMPORTS ===
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Sklearn - Preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Sklearn - Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor

# Sklearn - Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Sklearn - Feature Selection
from sklearn.feature_selection import mutual_info_regression

# XGBoost
from xgboost import XGBRegressor

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("✅ Libraries imported successfully")

# %% [markdown]
# ---
# ## 1. Data Loading & Initial Exploration
#
# **Goal:** Understand the dataset structure, dimensions, data types, and get initial insights.

# %%
# Load data
df = pd.read_csv("houses_Madrid.csv")

print(f"📊 Dataset shape: {df.shape[0]:,} samples × {df.shape[1]} features")
print(f"\n{'='*60}")
print("First 5 rows:")
df.head()

# %%
# Data types and memory usage
print("📋 Data Types Summary:")
print(f"  - Numeric features: {df.select_dtypes(include=[np.number]).shape[1]}")
print(f"  - Categorical features: {df.select_dtypes(include=['object']).shape[1]}")
print(f"  - Boolean features: {df.select_dtypes(include=['bool']).shape[1]}")
print(f"\n  - Total memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# %%
# Target variable statistics
print("🎯 Target Variable (buy_price) Statistics:")
print(f"  - Mean:   €{df['buy_price'].mean():,.0f}")
print(f"  - Median: €{df['buy_price'].median():,.0f}")
print(f"  - Std:    €{df['buy_price'].std():,.0f}")
print(f"  - Min:    €{df['buy_price'].min():,.0f}")
print(f"  - Max:    €{df['buy_price'].max():,.0f}")
print(f"  - Skewness: {df['buy_price'].skew():.2f} (>1 = highly right-skewed)")

# %% [markdown]
# ---
# ## 2. Data Quality Assessment
#
# ### Why this step is critical:
# Missing data can bias models and reduce statistical power. Before deciding how to handle it,
# we need to understand:
# - **How much** is missing? (<5% → can drop; 5-20% → impute; >50% → consider dropping the feature)
# - **What pattern?** (MCAR, MAR, MNAR — determines the best imputation strategy)
# - **Which features** are affected?

# %%
# Missing values analysis
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Feature': missing.index,
    'Missing Count': missing.values,
    'Missing %': missing_pct.values
}).query('`Missing Count` > 0').sort_values('Missing %', ascending=False)

print(f"🔍 Features with missing values: {len(missing_df)} of {df.shape[1]}")
print(f"\n{'Feature':<25} {'Missing Count':<15} {'Missing %':<10} {'Action'}")
print("─" * 70)
for _, row in missing_df.iterrows():
    pct = row['Missing %']
    if pct > 50:
        action = "❌ DROP FEATURE (>50% missing)"
    elif pct > 20:
        action = "⚠️ Impute with caution"
    elif pct > 5:
        action = "🔄 Impute (median/KNN)"
    else:
        action = "✅ Drop rows or simple impute"
    print(f"  {row['Feature']:<23} {int(row['Missing Count']):<15} {pct:<10.1f} {action}")

# %% [markdown]
# ### Decision Rationale for Missing Values:
#
# | Missing % | Strategy | Justification |
# |-----------|----------|---------------|
# | < 5% | Drop rows | Minimal information loss; statistically insignificant |
# | 5-20% | Median imputation | Robust to outliers; preserves central tendency |
# | 20-50% | Advanced imputation (KNN/MICE) | Too much data to discard; simple imputation distorts distribution |
# | > 50% | Drop the feature | Feature is unreliable; imputation would fabricate more data than it preserves |
#
# **Important:** `sq_mt_useful` has ~62% missing → we drop this feature entirely.
# Even though it may be highly correlated with price, imputing 62% of values would 
# essentially be "inventing" data, making any correlation artificial.

# %%
# Apply missing value strategy
# Step 1: Identify features to drop (>50% missing)
features_to_drop = missing_df[missing_df['Missing %'] > 50]['Feature'].tolist()
print(f"🗑️ Dropping features with >50% missing: {features_to_drop}")

# Step 2: Drop those features
df_clean = df.drop(columns=features_to_drop)

# Step 3: For remaining features, we'll handle missing values AFTER the train/test split
# to avoid data leakage (statistics calculated only on training data)
print(f"\n📊 Dataset after dropping high-missing features: {df_clean.shape}")

# %% [markdown]
# ---
# ## 3. Feature Selection
#
# ### Approaches used (from simple to sophisticated):
# 1. **Pearson Correlation** — Captures linear relationships (sensitive to outliers)
# 2. **Spearman Correlation** — Captures monotonic relationships (robust to outliers)
# 3. **Mutual Information** — Captures ANY dependency (linear + non-linear)
#
# ### Why multiple methods?
# Using only Pearson correlation (as done in basic tutorials) has limitations:
# - Assumes linear relationship between feature and target
# - Heavily influenced by outliers (uses mean and std)
# - Misses non-linear patterns (e.g., neighborhood effect on price)
#
# A professional approach cross-references multiple methods and uses domain knowledge.

# %%
# Select only numeric features for correlation analysis
numeric_df = df_clean.select_dtypes(include=[np.number])

# Remove rows where buy_price is null for correlation analysis
numeric_df = numeric_df.dropna(subset=['buy_price'])

print(f"📊 Numeric features available: {numeric_df.shape[1] - 1} (excluding target)")

# %%
# === METHOD 1: Pearson Correlation ===
# Measures LINEAR correlation between each feature and buy_price
# Range: [-1, 1] | Sensitive to outliers

pearson_corr = numeric_df.corr(method='pearson')['buy_price'].drop('buy_price').sort_values(ascending=False)

print("📈 PEARSON Correlation with buy_price (top 15):")
print("─" * 50)
for feat, corr in pearson_corr.head(15).items():
    bar = "█" * int(abs(corr) * 30)
    print(f"  {feat:<25} {corr:+.3f}  {bar}")

# %%
# === METHOD 2: Spearman Correlation ===
# Measures MONOTONIC correlation (based on ranks, not values)
# Range: [-1, 1] | Robust to outliers

spearman_corr = numeric_df.corr(method='spearman')['buy_price'].drop('buy_price').sort_values(ascending=False)

print("📈 SPEARMAN Correlation with buy_price (top 15):")
print("─" * 50)
for feat, corr in spearman_corr.head(15).items():
    bar = "█" * int(abs(corr) * 30)
    print(f"  {feat:<25} {corr:+.3f}  {bar}")

# %%
# === METHOD 3: Mutual Information ===
# Measures ANY statistical dependency (linear + non-linear)
# Range: [0, ∞) | Higher = more dependency
# Note: Requires no missing values, so we impute temporarily for this analysis only

from sklearn.impute import SimpleImputer

# Temporary imputation for MI calculation only
temp_imputer = SimpleImputer(strategy='median')
X_temp = numeric_df.drop(columns=['buy_price'])
X_temp_imputed = pd.DataFrame(
    temp_imputer.fit_transform(X_temp),
    columns=X_temp.columns
)
y_temp = numeric_df['buy_price'].values

mi_scores = mutual_info_regression(X_temp_imputed, y_temp, random_state=RANDOM_STATE)
mi_series = pd.Series(mi_scores, index=X_temp.columns).sort_values(ascending=False)

print("📈 MUTUAL INFORMATION with buy_price (top 15):")
print("─" * 50)
for feat, mi in mi_series.head(15).items():
    bar = "█" * int(mi * 15)
    print(f"  {feat:<25} {mi:.3f}  {bar}")

# %%
# === CROSS-REFERENCE: Comparing all methods ===
comparison = pd.DataFrame({
    'Pearson': pearson_corr,
    'Spearman': spearman_corr,
    'MI': mi_series
}).dropna()

# Rank each method
comparison['Pearson_Rank'] = comparison['Pearson'].abs().rank(ascending=False)
comparison['Spearman_Rank'] = comparison['Spearman'].abs().rank(ascending=False)
comparison['MI_Rank'] = comparison['MI'].rank(ascending=False)
comparison['Avg_Rank'] = comparison[['Pearson_Rank', 'Spearman_Rank', 'MI_Rank']].mean(axis=1)
comparison = comparison.sort_values('Avg_Rank')

print("🏆 FEATURE RANKING (Cross-referencing all methods):")
print("─" * 70)
print(f"  {'Feature':<25} {'Pearson':<10} {'Spearman':<10} {'MI':<8} {'Avg Rank'}")
print("─" * 70)
for feat, row in comparison.head(10).iterrows():
    print(f"  {feat:<25} {row['Pearson']:+.3f}    {row['Spearman']:+.3f}    {row['MI']:.3f}   {row['Avg_Rank']:.1f}")

# %% [markdown]
# ### Feature Selection Decision
#
# Based on the cross-reference analysis, we select features that consistently rank high
# across all three methods. This gives us confidence that the relationship is:
# - **Linear** (Pearson) ✓
# - **Monotonic** (Spearman) ✓
# - **Statistically dependent** (MI) ✓
#
# **Selected features:** `sq_mt_built`, `n_bathrooms`, `n_rooms`
#
# **Note on multicollinearity:** `sq_mt_built` and `n_bathrooms` have high inter-correlation (~0.84).
# This is acceptable for tree-based models but can inflate coefficients in linear models.
# We keep both because:
# 1. Tree models (our expected best performers) handle collinearity naturally
# 2. For linear models, we'll use Ridge/Lasso which handle it via regularization

# %%
# === CHECK MULTICOLLINEARITY ===
selected_features = ['sq_mt_built', 'n_bathrooms', 'n_rooms']

print("🔗 Multicollinearity Check (inter-feature correlations):")
print("─" * 50)
inter_corr = numeric_df[selected_features].corr()
print(inter_corr.to_string())
print("\n⚠️  sq_mt_built × n_bathrooms = {:.3f} (high collinearity)".format(
    inter_corr.loc['sq_mt_built', 'n_bathrooms']))
print("   → Acceptable for tree-based models")
print("   → Linear models should use regularization (Ridge/Lasso)")

# %%
# === PREPARE FINAL DATASET ===
# Keep only selected features + target, drop rows with ANY null in these columns

features = selected_features
target = 'buy_price'

df_model = df_clean[features + [target]].dropna()
print(f"📊 Dataset after selecting features and dropping nulls: {df_model.shape}")
print(f"   Rows removed: {len(df_clean) - len(df_model):,}")

# %% [markdown]
# ---
# ## 4. Train/Test Split — BEFORE Outlier Removal
#
# ### Critical: Why split FIRST?
#
# **Data Leakage occurs when:**
# - You calculate statistics (percentiles, mean, std) on ALL data including test
# - You remove outliers based on global thresholds
# - You normalize/scale using global parameters
#
# **The test set must simulate "unseen future data."** In production, you won't have access
# to future data to calculate percentiles. All decisions must be based ONLY on training data.
#
# **Correct order:** Split → Remove outliers from train → Fit scaler on train → Transform test

# %%
# === TRAIN/TEST SPLIT (80/20) ===
X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

print(f"📊 Train/Test Split:")
print(f"   Training set:  {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"   Test set:      {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.0f}%)")

# %% [markdown]
# ---
# ## 5. Outlier Detection & Treatment (on Training Data ONLY)
#
# ### Method: Percentile-based filtering (Q1/Q99)
#
# **Pros:**
# - Simple, interpretable, statistically grounded
# - Effective at removing extreme values that distort linear models
#
# **Cons:**
# - Arbitrary threshold — why Q99 and not Q97?
# - Removes real data points (expensive houses DO exist)
# - Creates model blind spots (can't predict high-end properties)
#
# **Alternative approaches (noted for future improvement):**
# - IQR method: outlier = value outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
# - Z-score: |z| > 3 is an outlier
# - Isolation Forest: ML-based anomaly detection
#
# **Decision:** We use Q1/Q99 (conservative) to remove only extreme cases
# while preserving the majority of the distribution. Applied ONLY on training data.

# %%
# === OUTLIER REMOVAL (TRAINING SET ONLY) ===
print("🔍 Outlier Analysis (Training Set):")
print("─" * 60)

# Calculate percentiles on TRAINING data only
outlier_lower = {}
outlier_upper = {}

for col in features + [target]:
    q01 = X_train[col].quantile(0.01) if col in features else y_train.quantile(0.01)
    q99 = X_train[col].quantile(0.99) if col in features else y_train.quantile(0.99)
    outlier_lower[col] = q01
    outlier_upper[col] = q99
    
    data = X_train[col] if col in features else y_train
    n_outliers = ((data < q01) | (data > q99)).sum()
    print(f"  {col:<15} Q01={q01:>10,.0f}  Q99={q99:>10,.0f}  Outliers: {n_outliers}")

# %%
# Apply outlier filter to training data
mask_train = pd.Series([True] * len(X_train), index=X_train.index)

for col in features:
    mask_train &= (X_train[col] >= outlier_lower[col]) & (X_train[col] <= outlier_upper[col])

# Also filter target
mask_train &= (y_train >= outlier_lower[target]) & (y_train <= outlier_upper[target])

X_train_clean = X_train[mask_train]
y_train_clean = y_train[mask_train]

removed = len(X_train) - len(X_train_clean)
print(f"\n✂️  Outlier Removal Results:")
print(f"   Before: {len(X_train):,} samples")
print(f"   After:  {len(X_train_clean):,} samples")
print(f"   Removed: {removed:,} ({removed/len(X_train)*100:.1f}%)")
print(f"\n⚠️  Test set remains UNTOUCHED ({len(X_test):,} samples)")
print(f"   → This simulates real-world conditions where we can't control incoming data")

# %% [markdown]
# ---
# ## 6. Model Training & Comparison
#
# ### Models selected and why:
#
# | Model | Type | Why include it? |
# |-------|------|-----------------|
# | DummyRegressor | Baseline | Always predict mean — any real model must beat this |
# | LinearRegression | Linear | Simple, interpretable, fast. Tests if relationship is linear |
# | Ridge | Linear + L2 reg | Handles multicollinearity (sq_mt × n_bathrooms) |
# | Lasso | Linear + L1 reg | Feature selection built-in — may zero out redundant features |
# | Random Forest | Ensemble (Bagging) | Captures non-linearity, robust to outliers |
# | Gradient Boosting | Ensemble (Boosting) | Sequential error correction, often best performer |
# | XGBoost | Ensemble (Boosting) | Optimized GB with regularization, state-of-the-art |
#
# ### Why use Pipeline?
# Pipeline ensures that StandardScaler is fit ONLY on training data and applied (transform only)
# to test data. This prevents data leakage at the preprocessing stage.

# %%
# === DEFINE MODELS ===
models = {
    "Dummy (Baseline)": Pipeline([
        ('scaler', StandardScaler()),
        ('model', DummyRegressor(strategy='mean'))
    ]),
    "Linear Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),
    "Ridge (L2)": Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ]),
    "Lasso (L1)": Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(alpha=1.0))
    ]),
    "ElasticNet": Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet(alpha=1.0, l1_ratio=0.5))
    ]),
    "Random Forest": Pipeline([
        ('scaler', StandardScaler()),  # RF doesn't need scaling, but keeps pipeline consistent
        ('model', RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=5, random_state=RANDOM_STATE
        ))
    ]),
    "Gradient Boosting": Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=RANDOM_STATE
        ))
    ]),
    "XGBoost": Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_STATE, verbosity=0
        ))
    ]),
}

print(f"🤖 Models to compare: {len(models)}")
for name in models:
    print(f"   • {name}")

# %% [markdown]
# ---
# ## 7. Cross-Validation (5-Fold)
#
# ### Why Cross-Validation instead of a single train/test evaluation?
#
# A single split can be "lucky" or "unlucky". Cross-validation:
# - Trains and evaluates the model on **5 different splits**
# - Every data point appears in the test set exactly once
# - Provides **mean ± std** of metrics → much more reliable estimate
# - The standard deviation tells us about model **stability**
#
# ```
# Fold 1: [TEST  | train | train | train | train]
# Fold 2: [train | TEST  | train | train | train]
# Fold 3: [train | train | TEST  | train | train]
# Fold 4: [train | train | train | TEST  | train]
# Fold 5: [train | train | train | train | TEST ]
# ```

# %%
# === CROSS-VALIDATION ON TRAINING DATA ===
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("📊 Cross-Validation Results (5-Fold on Training Data):")
print("═" * 75)
print(f"  {'Model':<22} {'R² (mean±std)':<20} {'MAE (mean±std)':<25} {'Status'}")
print("─" * 75)

cv_results = {}

for name, pipe in models.items():
    r2_scores = cross_val_score(pipe, X_train_clean, y_train_clean, cv=kf, scoring='r2')
    mae_scores = -cross_val_score(pipe, X_train_clean, y_train_clean, cv=kf, scoring='neg_mean_absolute_error')
    
    cv_results[name] = {
        'r2_mean': r2_scores.mean(),
        'r2_std': r2_scores.std(),
        'mae_mean': mae_scores.mean(),
        'mae_std': mae_scores.std()
    }
    
    status = "🏆" if r2_scores.mean() > 0.6 else "✅" if r2_scores.mean() > 0.3 else "⚠️"
    print(f"  {name:<22} {r2_scores.mean():.4f} ± {r2_scores.std():.3f}   "
          f"€{mae_scores.mean():>9,.0f} ± {mae_scores.std():>7,.0f}   {status}")

print("─" * 75)
print("  🏆 = R² > 0.6 | ✅ = R² > 0.3 | ⚠️ = Weak model")

# %% [markdown]
# ---
# ## 8. Train vs Test Error Analysis (Bias-Variance Diagnosis)
#
# ### Why compare train and test errors?
#
# | Train Error | Test Error | Diagnosis | Solution |
# |-------------|-----------|-----------|----------|
# | Low | Low | ✅ Good fit | — |
# | Low | High | ⚠️ Overfitting | More regularization, less complexity |
# | High | High | ⚠️ Underfitting | More features, more complexity |
# | High | Low | 🐛 Bug/Data issue | Check implementation |
#
# This comparison was **missing in the original notebook** — it's essential for model diagnostics.

# %%
# === TRAIN vs TEST ERROR (Full evaluation) ===
print("📊 Train vs Test Comparison (Bias-Variance Diagnosis):")
print("═" * 90)
print(f"  {'Model':<22} {'R² Train':<12} {'R² Test':<12} {'MAE Train':<14} {'MAE Test':<14} {'Diagnosis'}")
print("─" * 90)

final_results = {}

for name, pipe in models.items():
    # Fit on training data
    pipe.fit(X_train_clean, y_train_clean)
    
    # Predict on TRAIN
    y_hat_train = pipe.predict(X_train_clean)
    
    # Predict on TEST (untouched data!)
    y_hat_test = pipe.predict(X_test)
    
    # Metrics — NOTE: correct argument order is (y_true, y_pred)
    r2_train = r2_score(y_train_clean, y_hat_train)
    r2_test = r2_score(y_test, y_hat_test)
    mae_train = mean_absolute_error(y_train_clean, y_hat_train)
    mae_test = mean_absolute_error(y_test, y_hat_test)
    rmse_train = np.sqrt(mean_squared_error(y_train_clean, y_hat_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_hat_test))
    
    # Diagnosis
    gap = r2_train - r2_test
    if r2_train < 0.3:
        diagnosis = "❌ Underfitting"
    elif gap > 0.15:
        diagnosis = "⚠️ Overfitting"
    elif gap < 0.05 and r2_test > 0.5:
        diagnosis = "✅ Good fit"
    else:
        diagnosis = "🔄 Acceptable"
    
    final_results[name] = {
        'r2_train': r2_train, 'r2_test': r2_test,
        'mae_train': mae_train, 'mae_test': mae_test,
        'rmse_train': rmse_train, 'rmse_test': rmse_test,
        'diagnosis': diagnosis
    }
    
    print(f"  {name:<22} {r2_train:<12.4f} {r2_test:<12.4f} €{mae_train:<13,.0f} €{mae_test:<13,.0f} {diagnosis}")

print("─" * 90)
print("\n  📝 Interpretation:")
print("     • Gap(R² train - R² test) > 0.15 → Overfitting")
print("     • Both R² < 0.3 → Underfitting")
print("     • Small gap + high R² test → Good generalization")

# %% [markdown]
# ---
# ## 9. Hyperparameter Tuning (RandomizedSearchCV)
#
# ### Why tune hyperparameters?
# Default parameters rarely give the best performance. Tuning finds the optimal configuration
# for YOUR specific dataset.
#
# ### Why RandomizedSearchCV over GridSearchCV?
# - Grid Search tests ALL combinations (exponential growth with parameters)
# - Random Search samples N random combinations — empirically shown to be more efficient
#   (Bergstra & Bengio, 2012)
# - With 5-fold CV and 50 iterations, we evaluate 250 model fits (tractable)

# %%
# === HYPERPARAMETER TUNING: XGBoost ===
print("🔧 Tuning XGBoost hyperparameters...")
print("   Method: RandomizedSearchCV (50 iterations × 5 folds = 250 fits)")
print("   This may take a minute...\n")

param_distributions = {
    'model__n_estimators': [100, 200, 300, 500, 700, 1000],
    'model__max_depth': [3, 4, 5, 6, 7, 8, 10],
    'model__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
    'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'model__reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
    'model__reg_lambda': [0.5, 1.0, 2.0, 5.0],
    'model__min_child_weight': [1, 3, 5, 7],
}

xgb_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(random_state=RANDOM_STATE, verbosity=0))
])

random_search = RandomizedSearchCV(
    xgb_pipe,
    param_distributions=param_distributions,
    n_iter=50,
    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring='r2',
    random_state=RANDOM_STATE,
    n_jobs=-1,  # Use all CPU cores
    verbose=0
)

random_search.fit(X_train_clean, y_train_clean)

print(f"🏆 Best R² (CV): {random_search.best_score_:.4f}")
print(f"\n📋 Best Hyperparameters:")
for param, value in random_search.best_params_.items():
    print(f"   {param.replace('model__', ''):<20} = {value}")

# %%
# === EVALUATE TUNED MODEL ===
best_model = random_search.best_estimator_

y_hat_train_best = best_model.predict(X_train_clean)
y_hat_test_best = best_model.predict(X_test)

print("📊 Tuned XGBoost Results:")
print("─" * 50)
print(f"  {'Metric':<15} {'Train':<15} {'Test'}")
print("─" * 50)
print(f"  {'R²':<15} {r2_score(y_train_clean, y_hat_train_best):<15.4f} {r2_score(y_test, y_hat_test_best):.4f}")
print(f"  {'MAE':<15} €{mean_absolute_error(y_train_clean, y_hat_train_best):<14,.0f} €{mean_absolute_error(y_test, y_hat_test_best):,.0f}")
print(f"  {'RMSE':<15} €{np.sqrt(mean_squared_error(y_train_clean, y_hat_train_best)):<14,.0f} €{np.sqrt(mean_squared_error(y_test, y_hat_test_best)):,.0f}")

gap = r2_score(y_train_clean, y_hat_train_best) - r2_score(y_test, y_hat_test_best)
print(f"\n  Gap (R² train - test): {gap:.4f} {'✅ Good' if gap < 0.1 else '⚠️ Some overfitting'}")

# %% [markdown]
# ---
# ## 10. Feature Importance Analysis
#
# Understanding which features the model relies on most helps:
# - Validate that the model learned meaningful patterns
# - Guide future feature engineering
# - Provide business interpretability

# %%
# === FEATURE IMPORTANCE (from best model) ===
# Extract the XGBoost model from the pipeline
xgb_model = best_model.named_steps['model']

importances = pd.Series(
    xgb_model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("📊 Feature Importance (Tuned XGBoost):")
print("─" * 50)
for feat, imp in importances.items():
    bar = "█" * int(imp * 50)
    print(f"  {feat:<15} {imp:.3f}  {bar}")

print("\n📝 Interpretation:")
print("   Feature importance = how much each feature contributes")
print("   to reducing prediction error across all trees")

# %% [markdown]
# ---
# ## 11. Error Analysis — Understanding Where the Model Fails
#
# A good data scientist doesn't just report metrics — they investigate WHERE and WHY
# the model makes mistakes.

# %%
# === RESIDUAL ANALYSIS ===
residuals = y_test - y_hat_test_best

print("📊 Residual Analysis (Test Set):")
print("─" * 50)
print(f"  Mean residual:   €{residuals.mean():,.0f} (should be ~0)")
print(f"  Std residual:    €{residuals.std():,.0f}")
print(f"  Median residual: €{residuals.median():,.0f}")
print(f"  Min residual:    €{residuals.min():,.0f} (worst under-prediction)")
print(f"  Max residual:    €{residuals.max():,.0f} (worst over-prediction)")

# Distribution of absolute errors
abs_errors = np.abs(residuals)
print(f"\n  Predictions within €50k:  {(abs_errors < 50000).mean()*100:.1f}%")
print(f"  Predictions within €100k: {(abs_errors < 100000).mean()*100:.1f}%")
print(f"  Predictions within €150k: {(abs_errors < 150000).mean()*100:.1f}%")

# %%
# === WHERE DOES THE MODEL FAIL? ===
# Analyze errors by price segment
test_analysis = pd.DataFrame({
    'actual_price': y_test.values,
    'predicted_price': y_hat_test_best,
    'abs_error': abs_errors.values
})

# Create price segments
bins = [0, 200000, 400000, 600000, 800000, float('inf')]
labels = ['<200k', '200-400k', '400-600k', '600-800k', '>800k']
test_analysis['segment'] = pd.cut(test_analysis['actual_price'], bins=bins, labels=labels)

print("📊 Error by Price Segment:")
print("─" * 60)
print(f"  {'Segment':<12} {'Count':<8} {'MAE':<15} {'Mean Actual':<15} {'Error %'}")
print("─" * 60)

for segment in labels:
    subset = test_analysis[test_analysis['segment'] == segment]
    if len(subset) > 0:
        mae_seg = subset['abs_error'].mean()
        mean_price = subset['actual_price'].mean()
        error_pct = mae_seg / mean_price * 100
        print(f"  {segment:<12} {len(subset):<8} €{mae_seg:<14,.0f} €{mean_price:<14,.0f} {error_pct:.1f}%")

print("\n📝 Note: Higher price segments typically have larger absolute errors")
print("   but the relative error (%) reveals where the model truly struggles.")

# %% [markdown]
# ---
# ## 12. Final Summary & Conclusions
#
# ### Key Findings

# %%
# === FINAL SUMMARY ===
print("=" * 70)
print("                    📋 FINAL SUMMARY")
print("=" * 70)

print(f"""
📊 Dataset:
   • Original: 21,742 samples × 58 features
   • After cleaning: {len(df_model):,} samples × {len(features)} features + target
   • Train: {len(X_train_clean):,} | Test: {len(X_test):,}

🎯 Features Selected: {', '.join(features)}
   • Method: Cross-referenced Pearson, Spearman, and Mutual Information

🤖 Best Model: Tuned XGBoost
   • R² (test): {r2_score(y_test, y_hat_test_best):.4f}
   • MAE (test): €{mean_absolute_error(y_test, y_hat_test_best):,.0f}
   • RMSE (test): €{np.sqrt(mean_squared_error(y_test, y_hat_test_best)):,.0f}

📈 Model Ranking (R² test):
""")

# Sort and display all models
all_test_r2 = {name: final_results[name]['r2_test'] for name in final_results}
all_test_r2['XGBoost (Tuned)'] = r2_score(y_test, y_hat_test_best)

for rank, (name, r2) in enumerate(sorted(all_test_r2.items(), key=lambda x: x[1], reverse=True), 1):
    bar = "█" * int(r2 * 30) if r2 > 0 else ""
    print(f"   {rank}. {name:<25} R²={r2:.4f}  {bar}")

# %% [markdown]
# ---
# ## 13. Recommendations for Improvement
#
# ### What would push R² from ~0.65 to ~0.90?
#
# 1. **Include categorical features** — `neighborhood_id`, `house_type_id`, `has_lift`, `is_exterior`
#    are extremely relevant for price. Location is everything in real estate.
#
# 2. **Feature Engineering:**
#    - Price per m² by neighborhood (target encoding)
#    - Distance to metro/city center
#    - Floor × has_lift interaction
#    - Age of building (if available)
#
# 3. **More advanced models:**
#    - LightGBM (faster, often better with categorical features)
#    - CatBoost (native categorical handling)
#    - Stacking ensemble (combine multiple models)
#
# 4. **Target transformation:**
#    - Log-transform `buy_price` (reduces skewness)
#    - Helps linear models capture multiplicative relationships
#
# 5. **Spatial features:**
#    - Latitude/Longitude (if available)
#    - Geospatial clustering
#
# ---
#
# ## Key Takeaways (What I Learned)
#
# | Topic | Key Insight |
# |-------|-------------|
# | Data Leakage | Always split BEFORE preprocessing |
# | Feature Selection | Use multiple methods, not just Pearson |
# | Outlier Treatment | Apply only on training data |
# | Model Evaluation | Always compare train vs test error |
# | Cross-Validation | Single split is unreliable |
# | Hyperparameter Tuning | Defaults are rarely optimal |
# | Pipeline | Ensures reproducibility and prevents leakage |
#
# ---
#
# ## How to Convert This to .ipynb
#
# ```bash
# # Option 1: Using jupytext
# pip install jupytext
# jupytext --to notebook madrid_house_prices_notebook.py
#
# # Option 2: Open in VS Code with Jupyter extension
# # The # %% markers are automatically recognized as cells
# ```

