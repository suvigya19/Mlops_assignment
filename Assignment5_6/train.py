import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('/Users/suvigyasharma/Mlops_assignment/Assignment5_6/data/hour.csv')

# Inspect for any missing values
print(df.isnull().sum())

# Feature Engineering
# Create a new column "day_night" to differentiate between day and night hours
df['day_night'] = df['hr'].apply(lambda x: 'day' if 6 <= x <= 18 else 'night')

# Drop unnecessary columns
df.drop(['instant', 'casual', 'registered'], axis=1, inplace=True)

# Convert 'dteday' to a datetime object
df['dteday'] = pd.to_datetime(df['dteday'])

# Convert categorical columns to 'category' dtype
categorical_columns = ['season', 'holiday', 'weekday', 'weathersit', 'workingday', 'mnth', 'yr', 'hr']
for col in categorical_columns:
    df[col] = df[col].astype('category')

# Remove 'dteday' since it's no longer needed
df.drop(columns=['dteday'], inplace=True)

# Separate features from the target
X = df.drop(columns=['cnt'])
y = df['cnt']

# Numerical Preprocessing Pipeline
num_features = ['temp', 'hum', 'windspeed']
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
    ('scaler', MinMaxScaler())  # Scale features between 0 and 1
])

# Apply transformation to numerical features
X[num_features] = num_pipeline.fit_transform(X[num_features])

# Categorical Preprocessing Pipeline
cat_features = ['season', 'weathersit', 'day_night']
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with most frequent value
    ('onehot', OneHotEncoder(sparse_output=False, drop='first'))  # Apply one-hot encoding, drop first level
])

# Apply transformation to categorical features
X_encoded = cat_pipeline.fit_transform(X[cat_features])

# Convert the encoded categorical data to a DataFrame
X_encoded = pd.DataFrame(X_encoded, columns=cat_pipeline.named_steps['onehot'].get_feature_names_out(cat_features))

# Concatenate numerical and encoded categorical features
X = pd.concat([X.drop(columns=cat_features), X_encoded], axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Linear Regression Model Training and Logging
def train_linear_regression():
    with mlflow.start_run(run_name="Linear_Regression") as run:
        # Train the model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        # Predict on test set
        predictions = lr_model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Log metrics and parameters in MLflow
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.log_param("model", "Linear Regression")

        # Log the model to MLflow
        mlflow.sklearn.log_model(lr_model, "model")

        # Output metrics
        print(f"Linear Regression MSE: {mse}")
        print(f"Linear Regression R²: {r2}")
        
        # Return MSE and the run_id for this model
        return mse, run.info.run_id


# Random Forest Model Training and Logging
def train_random_forest():
    with mlflow.start_run(run_name="Random_Forest") as run:
        # Train the model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Predict on test set
        predictions = rf_model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Log metrics and parameters in MLflow
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.log_param("model", "Random Forest")

        # Log the model to MLflow
        mlflow.sklearn.log_model(rf_model, "model")

        # Output metrics
        print(f"Random Forest MSE: {mse}")
        print(f"Random Forest R²: {r2}")
        
        # Return MSE and the run_id for this model
        return mse, run.info.run_id


# Save the best performing model to the MLflow Model Registry
def register_best_model(mse_lr, mse_rf, run_id_lr, run_id_rf):
    if mse_lr < mse_rf:
        best_model_run_id = run_id_lr
        best_model_name = "Linear_Regression"
    else:
        best_model_run_id = run_id_rf
        best_model_name = "Random_Forest"
    
    # Register the best model to the Model Registry
    print(f"Best model is {best_model_name} with run id {best_model_run_id}")
    mlflow.register_model(f"runs:/{best_model_run_id}/model", best_model_name)


if __name__ == "__main__":
    # Train both models and get their MSE and run_id
    mse_lr, run_id_lr = train_linear_regression()
    mse_rf, run_id_rf = train_random_forest()

    # Compare the models and register the best one in the Model Registry
    register_best_model(mse_lr, mse_rf, run_id_lr, run_id_rf)
