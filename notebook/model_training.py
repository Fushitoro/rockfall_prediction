# Step 1: Import necessary libraries
import warnings

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings('ignore')

print("=== STEP 1: LIBRARY IMPORTS COMPLETE ===")
print("Libraries loaded successfully!")