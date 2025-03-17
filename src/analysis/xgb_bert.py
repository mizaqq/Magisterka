import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error
import xgboost as xgb
from sentence_transformers import SentenceTransformer

# Load data
df = pd.read_csv('/home/miza/Magisterka/src/data/gpt_generated_data.csv')

# Encode labels
label_encoder = LabelEncoder()
df['Category_encoded'] = label_encoder.fit_transform(df['Category'])

# Extract BERT embeddings
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
embeddings = model.encode(df['OCR_product'].tolist(), batch_size=32, show_progress_bar=True)

# Prepare data
X = embeddings = embeddings = embeddings = embeddings = torch.tensor(embeddings).numpy()
y_category = df['Category_encoded'].values
y_cost = df['Correct_cost'].values

# Split dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_cat_train, y_cat_test, y_cost_train, y_cost_test = train_test_split(
    X, y_category, y_cost, test_size=0.2, random_state=42
)

# Train XGBoost classifier for category
import xgboost as xgb

category_model = xgb.XGBClassifier(n_estimators=100)
category_model = category_model.fit(X_train, y_cat_train)

# Predict and evaluate category
y_cat_pred = category_model.predict(X_test)
from sklearn.metrics import classification_report, mean_squared_error

print("Classification Report (Category):")
print(classification_report(y_cat_test, y_cat_pred, target_names=label_encoder.classes_))

# Train regression model for cost
cost_model = xgb.XGBRegressor(n_estimators=100)
cost_model = cost_model.fit(X_train, y_cost_train)

# Predict and evaluate cost
y_cost_pred = cost_model.predict(X_test)
print("Mean Squared Error (Cost):", mean_squared_error(y_cost_test, y_cost_pred))

# Inference Example
sample_text = ["Ryż bialu 1kg 0 x4.29 34.32"]
sample_embedding = model.encode(sample_text)
predicted_category = category_model.predict(sample_embedding)[0]
predicted_category_label = label_encoder.inverse_transform([predicted_category])[0]
predicted_cost = cost_model.predict(sample_embedding)[0]

print(f"Predicted Category: {predicted_category}, Predicted Cost: {predicted_cost:.2f}")
