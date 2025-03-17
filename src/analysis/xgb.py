import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error
import xgboost as xgb

df = pd.read_csv('/home/miza/Magisterka/src/data/gpt_generated_data.csv')

label_encoder = LabelEncoder()
df['Category_encoded'] = label_encoder.fit_transform(df['Category'])

X_text = df['OCR_product']
y_category = df['Category_encoded']
y_cost = df['Correct_cost']

vectorizer = TfidfVectorizer(max_features=500)
X_tfidf = vectorizer.fit_transform(X=df['OCR_product'])

X_train, X_test, y_cat_train, y_cat_test, y_cost_train, y_cost_test = train_test_split(
    X_tfidf, y_category, y_cost, test_size=0.2, random_state=42
)

category_model = xgb.XGBClassifier(n_estimators=100, tree_method='gpu_hist')
category_model = category_model.fit(X_train, y_cat_train)

y_cat_pred = category_model.predict(X_test)
print("Classification Report (Category):")
print(classification_report(y_cat_test, y_cat_pred, target_names=label_encoder.classes_))

cost_model = xgb.XGBRegressor(n_estimators=100, tree_method='gpu_hist')
cost_model = cost_model.fit(X_train, y_cost_train)

y_cost_pred = cost_model.predict(X_test)
print("Mean Squared Error (Cost):", mean_squared_error(y_cost_test, y_cost_pred))

sample_text = ["Ryż bialu 1kg 0 x4.29 34.32"]
sample_tfidf = vectorizer.transform(sample_text)
predicted_category_id = category_model.predict(sample_tfidf)[0]
predicted_category = label_encoder.inverse_transform([predicted_category_id])[0]
predicted_cost = cost_model.predict(sample_tfidf)[0]

print(f"Predicted Category: {predicted_category}, Predicted Cost: {predicted_cost:.2f}")
