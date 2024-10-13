import pandas as pd
import numpy as np

df = pd.read_csv('new_preprocessed_loan_data.csv')
df.head()

df.dropna(inplace=True) # dropping any null values that present in the dataset

df['Verification_Status'].unique() # returns the unique values in the column

df['Verification_Status'] = df['Verification_Status'].replace('Source Verified', 'Verified')

df['Home_Ownership'].value_counts() # returns a Series containing counts of unique values of the column.

# Filter the DataFrame to exclude rows where the 'Home_Ownership' column has the value 'NONE'
df = df[df['Home_Ownership'] != 'NONE']
# Further filter the DataFrame to exclude rows where the 'Home_Ownership' column has the value 'ANY'
df = df[df['Home_Ownership'] != 'ANY']

df.drop(columns=['Unnamed: 0'], axis=1, inplace=True) # dropping the column

df['Terms(Months)'].value_counts() # returns a Series containing counts of unique values of the column.

columns_to_drop = ['Delinquency_2yrs', 'Total_Accounts','Sub_Grade',
       'Total_Received_Principal', 'Total_Received_Interest',
                   'Loan_Status', 'Issue_Month', 'Issue_Year', 'region',
                   'Public_Record', 'Annual_Income']
df = df.drop(columns=columns_to_drop)

df.columns # checking the columns of the updated dataframe

df.info() # prints information about a DataFrame including the index dtype and columns, non-null values and memory usage.

df['Purpose'].unique() # returns all the unique values in

nominal_columns = ['Purpose']
df = pd.get_dummies(df, columns=nominal_columns)



from sklearn.preprocessing import LabelEncoder

le_verification_status = LabelEncoder()
df['Verification_Status'] = le_verification_status.fit_transform(df['Verification_Status'])

le_grade = LabelEncoder()
df['Grade'] = le_grade.fit_transform(df['Grade'])
le_home_ownership=LabelEncoder()
df['Home_Ownership'] = le_home_ownership.fit_transform(df['Home_Ownership'])

verification_status_mapping = dict(zip(le_verification_status.classes_, le_verification_status.transform(le_verification_status.classes_)))
grade_mapping = dict(zip(le_grade.classes_, le_grade.transform(le_grade.classes_)))
home_ownership_mapping = dict(zip(le_home_ownership.classes_, le_home_ownership.transform(le_home_ownership.classes_)))

# print("Verification Status Mapping:", verification_status_mapping)
# print("Grade Mapping:", grade_mapping)
# print("Home Ownership Mapping:", home_ownership_mapping)

df_new=df.copy() # making a copy of the dataframe

df_new['Risk_Category'].value_counts() # counting the occurences of each unique value in the column

risk_mapping = {
    'High Risk': 2,
    'Moderate Risk': 1,
    'Low Risk': 0
}
df_new['Risk_Category'] = df_new['Risk_Category'].map(risk_mapping)
df_new['Risk_Category'].value_counts()



# import seaborn as sns
import matplotlib.pyplot as plt

# sns.boxplot(df_new['Interest_Rate'])
# plt.show()

def remove_outliers_iqr(df_new, columns):

    for column in columns:
        Q1 = df_new[column].quantile(0.25)
        Q3 = df_new[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_new = df_new[(df_new[column] >= lower_bound) & (df_new[column] <= upper_bound)]
    return df_new

columns_to_check = ['Loan_Amount', 'Interest_Rate','Employment_Length']
df_new = remove_outliers_iqr(df_new, columns_to_check)

# sns.boxplot(df_new['Interest_Rate'],whis=1.5)
# plt.show()


# Compute the correlation matrix
correlation_matrix = df_new.corr()

# Extract the correlation of each feature with the 'Risk_Category'
risk_category_corr = correlation_matrix[['Risk_Category']]

# Plot the heatmap
plt.figure(figsize=(10, 8))
# sns.heatmap(risk_category_corr, annot=True, cmap='coolwarm', cbar=True, linewidths=0.5)
plt.title('Correlation Heatmap with Risk Category')
# plt.show()


feature_columns = df_new.columns[df_new.columns != 'Risk_Category'] # all columns that are not Risk_Category
target_column = 'Risk_Category'
# X = df_new[feature_columns]
# y = df_new[target_column]
X = df_new.iloc[:, :-1] 
y = df_new.iloc[:, -1] 
print(X.columns)



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestClassifier
# Create an instance of the RandomForestClassifier
clf = RandomForestClassifier(random_state=42) # random_state=42 is used to ensure reproducibility by setting a seed for random number generation
# Fit the RandomForestClassifier model to the training data
clf.fit(X_train, y_train)

# Use the trained RandomForestClassifier model to make predictions on the test data
y_pred = clf.predict(X_test) # the predicted class labels for each sample in X_test are being stored in y_pred


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(cm)

# sns.heatmap(cm,annot=True).set(title='Confusion Matrix', xlabel='Actual', ylabel='Predicted')
# plt.show()


from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred)
# print('Classification Report:')
# print(report)


from sklearn.linear_model import LogisticRegression

# Define a grid of hyperparameters to search over
# 'C' is the regularization strength, with lower values indicating stronger regularization
# 'penalty' specifies the type of regularization to use ('none' for no regularization, 'l2' for L2 regularization)
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['None', 'l2'],
}

# Initialize the LogisticRegression model
log_reg = LogisticRegression(random_state=42, max_iter=1000)

from sklearn.model_selection import GridSearchCV

# Initialize GridSearchCV to perform an exhaustive search over the parameter grid
# 'scoring' specifies the metric to optimize (accuracy in this case)
# 'cv' specifies the number of cross-validation folds
# 'n_jobs=-1' allows parallel processing using all available CPU cores
# 'verbose=3' provides detailed output about the progress of the grid search
grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid,scoring='accuracy', cv=2,n_jobs=-1,verbose=3)

# Fit GridSearchCV on the training data to search for the best hyperparameters
grid_search.fit(X_train, y_train)

# Retrieve the best hyperparameters found during the grid search
best_params = grid_search.best_params_
# Retrieve the best score achieved with the best hyperparameters
best_score = grid_search.best_score_


# Use the best model to make predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

from imblearn.over_sampling import SMOTE

# Initialize the SMOTE object
smote = SMOTE()

# Apply SMOTE to the scaled feature set (X_scaled) and the target variable (y)
# This will create a balanced dataset by oversampling the minority class
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the resampled data into training and testing sets
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_resampled, y_resampled, train_size=0.7,test_size=0.3, random_state=42)

from xgboost import XGBClassifier

# Initialize the XGBClassifier
xgb_clf = XGBClassifier(random_state=42, eval_metric='mlogloss', n_estimators=100)

# eval_metric='mlogloss' specifies the evaluation metric to use, in this case, log loss for multi-class classification
# n_estimators=100 sets the number of boosting rounds (trees) to 100

# Train the XGBClassifier on the resampled training data
xgb_clf.fit(X_train_resampled, y_train_resampled)

# use the model to make predictions on the resampled test data
y_pred_xgb = xgb_clf.predict(X_test_resampled)

# Initialize an XGBClassifier instance for the final model
xgb_clf_final = XGBClassifier(random_state=42, eval_metric='mlogloss', n_estimators=100)

# Train the final XGBClassifier on the entire resampled dataset
xgb_clf_final.fit(X_resampled, y_resampled)



# Import the joblib library for saving and loading Python objects
import joblib

# Define the file path where the model will be saved
model_path = 'xgb_model.pkl'

# Save the trained XGBClassifier model to the specified file path using joblib
# joblib.dump serializes the model object and writes it to a file
joblib.dump(xgb_clf_final, model_path)

# pickling the model 
# import pickle 
# pickle_out = open("xgb_clf_final.pkl", "wb") 
# pickle.dump(xgb_clf_final, pickle_out) 
# pickle_out.close()

