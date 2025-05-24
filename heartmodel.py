import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, GridSearchCV, 
                                   StratifiedKFold, cross_val_score)
from sklearn.ensemble import (RandomForestClassifier, 
                            GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                           accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
import joblib 
import warnings
warnings.filterwarnings('ignore')

#Loading and validating the heart disease dataset
def load_and_validate_data(filepath):
    try:
        df = pd.read_csv(filepath)
        if 'target' not in df.columns:
            raise ValueError("Dataset must contain 'target' column")
        
        for col in df.columns:
            if col != 'target':
# Checking  for columns that accurately predict target
                if df.groupby(col)['target'].nunique().max() == 1:
                    print(f"Warning: Column '{col}' perfectly predicts target - removing")
                    df = df.drop(col, axis=1)
                elif df[col].nunique() == len(df):
                    print(f"Warning: Column '{col}' appears to be an ID - removing")
                    df = df.drop(col, axis=1)
                    
        return df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

# Loading  the dataset obtained from kaggle
try:
    df = load_and_validate_data('heart_cleaned.csv')
except Exception as e:
    print(f"Failed to load data: {str(e)}")
    exit()

## Displaying an overview of the dataset and summary statistics.

print("="*80)
print("Initial Data Analysis")
print("="*80)
print(f"Dataset shape: {df.shape}")
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())

# # Creating a bar plot to visualize the distribution of the target variable (showing presence of heart disease)
# saving the plot as 'heart disease target_distribution.png' with tight bounding box adjustments. for better clarity.

plt.figure(figsize=(10, 6))
target_counts = df['target'].value_counts()
sns.barplot(x=target_counts.index, y=target_counts.values)
plt.title('Target Variable Distribution')
plt.xlabel('Heart Disease (0=No, 1=Yes)')
plt.ylabel('Count')
plt.savefig('heart disease target_distribution.png', bbox_inches='tight') 
plt.close()

## Generating a heatmap to visualize the correlation between features
# The correlation values are displayed with two decimal precision using a 'coolwarm' color scheme.clarity
# The plot is saved as 'heart disease correlation_matrix.png' with a tight layout to ensure readability.

plt.figure(figsize=(14, 12))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
           cmap='coolwarm', center=0, square=True)
plt.title("Feature Correlation Matrix (Upper Triangle)")
plt.tight_layout()
plt.savefig('heart disease correlation_matrix.png', bbox_inches='tight')
plt.close()

# Separating features and target
X = df.drop('target', axis=1)
y = df['target']
feature_names = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance 
class_ratio = len(y[y==1]) / len(y)
if class_ratio < 0.4 or class_ratio > 0.6:
    print(f"\nClass imbalance detected (ratio: {class_ratio:.2f}), applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
else:
    print("\nNo significant class imbalance detected")
    X_resampled, y_resampled = X_scaled, y
#handling stability
print("\n" + "="*80)
print("Model Stability Validation Across Different Splits")
print("="*80)

# Testing Model stability
stability_results = []
for random_state in [42, 123, 456, 789, 101112]:
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled,
        test_size=0.2,
        random_state=random_state,
        stratify=y_resampled
    )
    
    #handling the  Feature selection
    rf = RandomForestClassifier(random_state=42)
    rfecv = RFECV(
        estimator=rf,
        step=1,
        cv=StratifiedKFold(3),
        scoring='accuracy',
        min_features_to_select=5
    )
    rfecv.fit(X_train, y_train)
    
    X_train_selected = rfecv.transform(X_train)
    X_test_selected = rfecv.transform(X_test)
    
    # Train and evaluate a simple model to check accuracy
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train_selected, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test_selected))
    
    stability_results.append(test_acc)
    print(f"Random state {random_state}: Test accuracy = {test_acc:.4f}")
#this checks stability
stability_range = max(stability_results) - min(stability_results)
print(f"\nAccuracy range across splits: {stability_range:.4f}")
if stability_range > 0.1:
    print("Warning: High variability across splits - potential data issues!")
else:
    print("Model performance is stable across different splits")

print("\n" + "="*80)
print("Final Model Training Setup")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,
    random_state=42,
    stratify=y_resampled
)

rfecv = RFECV(
    estimator=RandomForestClassifier(random_state=42),
    step=1,
    cv=StratifiedKFold(5),
    scoring='accuracy',
    min_features_to_select=5
)
rfecv.fit(X_train, y_train)

X_train_selected = rfecv.transform(X_train)
X_test_selected = rfecv.transform(X_test)
selected_feature_names = np.array(feature_names)[rfecv.support_]

print("\nSelected Features:")
print(selected_feature_names)
print(f"\nOptimal number of features: {rfecv.n_features_}")

#TRAINING THE MODELS
models = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42, solver='liblinear'),
        'params': {
            'C': np.logspace(-3, 2, 6),
            'penalty': ['l1', 'l2']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss'),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    },
    'Neural Network': {
        'model': MLPClassifier(random_state=42, early_stopping=True),
        'params': {
            'hidden_layer_sizes': [(50,), (100,)],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant', 'adaptive']
        }
    }
}

#Storing the results
results = {}
best_models = {}

print("\n" + "="*80)
print("Model Training and Evaluation")
print("="*80)

for name, config in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}")
    print(f"{'='*50}")
    
    #doing grid search with cross-validation
    try:
        grid = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid.fit(X_train_selected, y_train)
        
        # Storing the best model
        best_model = grid.best_estimator_
        best_models[name] = best_model
        
     
        cv_scores = cross_val_score(
            best_model,
            X_train_selected,
            y_train,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1
        )
       
        train_pred = best_model.predict(X_train_selected)
        y_pred = best_model.predict(X_test_selected)
        
        # checking probabilities
        if hasattr(best_model, "predict_proba"):
            y_pred_proba = best_model.predict_proba(X_test_selected)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            y_pred_proba = None
            roc_auc = None
        
        # Calculating the metrics
        metrics = {
            'model': best_model,
            'best_params': grid.best_params_,
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc,
            'cv_mean_accuracy': np.mean(cv_scores),
            'cv_std_accuracy': np.std(cv_scores),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        results[name] = metrics
        
        # output
        print(f"\n{name} Results:")
        print(f"Best Parameters: {grid.best_params_}")
        print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"CV Accuracy: {metrics['cv_mean_accuracy']:.4f} ± {metrics['cv_std_accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        if metrics['roc_auc'] is not None:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
    except Exception as e:
        print(f"\nError training {name}: {str(e)}")
        continue

#selecting the model
print("\n" + "="*80)
print("Model Comparison and Selection")
print("="*80)

# Create the results dataframe
results_df = pd.DataFrame(results).T

# Adding model complexity 
def get_model_complexity(model):
    if hasattr(model, 'n_estimators'):
        return model.n_estimators
    elif hasattr(model, 'hidden_layer_sizes'):
        return sum(model.hidden_layer_sizes)
    elif hasattr(model, 'coef_'):
        return np.count_nonzero(model.coef_)
    else:
        return 1

results_df['complexity'] = results_df['model'].apply(get_model_complexity)

#  balancing accuracy and complexity
results_df['score'] = (0.7 * results_df['test_accuracy'] + 
                      0.2 * results_df['cv_mean_accuracy'] + 
                      0.1 * (1 - results_df['complexity']/results_df['complexity'].max()))

# Sort by composite score computed above
results_df = results_df.sort_values('score', ascending=False)

print("\nModel Performance Comparison:")
print(results_df[['test_accuracy', 'cv_mean_accuracy', 'complexity', 'score']])

# choosing the best model
best_model_name = results_df.index[0]
best_model = results[best_model_name]['model']
print(f"\nSelected Best Model: {best_model_name}")
print(f"Composite Score: {results_df.loc[best_model_name, 'score']:.4f}")

#interpreting the models
print("\n" + "="*80)
print("Model Interpretation")
print("="*80)

# Doing Feature Importance
if hasattr(best_model, 'feature_importances_'):
    print("\nGenerating Feature Importance Plot...")
    importance = best_model.feature_importances_
    sorted_idx = np.argsort(importance)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance[sorted_idx], y=selected_feature_names[sorted_idx])
    plt.title(f'{best_model_name} - Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png', bbox_inches='tight')
    plt.close()
    print("Saved feature_importance.png")

elif hasattr(best_model, 'coef_'):
    print("\nGenerating Coefficient Plot...")
    coef = best_model.coef_[0]
    sorted_idx = np.argsort(np.abs(coef))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=coef[sorted_idx], y=selected_feature_names[sorted_idx])
    plt.title(f'{best_model_name} - Feature Coefficients')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('feature_coefficients.png', bbox_inches='tight')
    plt.close()
    print("Saved feature_coefficients.png")

# Doing Permutation Importance
print("\nCalculating Permutation Importance...")
perm_importance = permutation_importance(
    best_model,
    X_test_selected,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

sorted_idx = perm_importance.importances_mean.argsort()

plt.figure(figsize=(10, 6))
plt.barh(
    np.array(selected_feature_names)[sorted_idx],
    perm_importance.importances_mean[sorted_idx],
    xerr=perm_importance.importances_std[sorted_idx]
)
plt.title('Permutation Importance (Test Set)')
plt.xlabel('Mean Accuracy Decrease')
plt.tight_layout()
plt.savefig('permutation_importance.png', bbox_inches='tight')
plt.close()
print("Saved permutation_importance.png")

# SHAP Explanations for easier interpretability
if hasattr(best_model, 'predict_proba'):
    print("\nGenerating SHAP Explanations...")
    try:
        if hasattr(best_model, 'feature_importances_'):
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test_selected)
        else:
            explainer = shap.KernelExplainer(best_model.predict_proba, 
                                           X_train_selected[:100])
            shap_values = explainer.shap_values(X_test_selected[:100])
        
        plt.figure()
        shap.summary_plot(shap_values, X_test_selected, 
                         feature_names=selected_feature_names,
                         show=False)
        plt.title('SHAP Summary Plot')
        plt.tight_layout()
        plt.savefig('shap_summary.png', bbox_inches='tight')
        plt.close()
        print("Saved shap_summary.png")
        
    except Exception as e:
        print(f"Could not generate SHAP plot: {str(e)}")

# LIME Explanations also
print("\nGenerating LIME Explanation...")
try:
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_selected,
        mode="classification",
        feature_names=selected_feature_names,
        class_names=["No Heart Disease", "Heart Disease"],
        discretize_continuous=True,
        random_state=42
    )

    exp = lime_explainer.explain_instance(
        X_test_selected[0], 
        best_model.predict_proba,
        num_features=len(selected_feature_names)
    )
    
    fig = exp.as_pyplot_figure()
    plt.title('LIME Explanation for First Test Instance')
    plt.tight_layout()
    plt.savefig('lime_explanation.png', bbox_inches='tight')
    plt.close()
    print("Saved lime_explanation.png")
    
except Exception as e:
    print(f"Could not generate LIME explanation: {str(e)}")

# Doing Evaluation
print("\n" + "="*80)
print("Final Evaluation and Model Saving")
print("="*80)

y_pred = best_model.predict(X_test_selected)
y_pred_proba = best_model.predict_proba(X_test_selected)[:, 1] if hasattr(best_model, "predict_proba") else None

print("\nFinal Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
if y_pred_proba is not None:
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model charts and features
print("\nSaving model artifacts...")
try:
    joblib.dump(best_model, 'best_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(rfecv, 'feature_selector.joblib')
    results_df.to_csv('model_comparison.csv', index=True)
    
    print("\nSuccessfully saved:")
    print("- best_model.joblib (trained model)")
    print("- scaler.joblib (feature scaler)")
    print("- feature_selector.joblib (feature selector)")
    print("- model_comparison.csv (performance metrics)")
    print("- Various visualization plots (*.png)")
    
except Exception as e:
    print(f"\nError saving artifacts: {str(e)}")

print("\n" + "="*80)
print("Model Training Complete!")
print("="*80)