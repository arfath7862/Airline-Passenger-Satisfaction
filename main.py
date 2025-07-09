import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("airline_passenger_satisfaction_high_accuracy.csv")

# Show first 5 rows
print(df.head())

# ----------------------------------------
# STEP 1: Exploratory Data Analysis (EDA)
# ----------------------------------------

# Gender Distribution
sns.countplot(x='Gender', data=df)
plt.title("Passenger Gender Distribution")
plt.show()

# Satisfaction by Class
sns.countplot(x='Class', hue='satisfaction', data=df)
plt.title("Satisfaction by Class")
plt.show()

# Satisfaction by Type of Travel
sns.countplot(x='Type of Travel', hue='satisfaction', data=df)
plt.title("Satisfaction by Type of Travel")
plt.show()

# Correlation Heatmap (after encoding)
df_encoded = df.copy()
label_enc = LabelEncoder()
df_encoded['Gender'] = label_enc.fit_transform(df_encoded['Gender'])
df_encoded['Customer Type'] = label_enc.fit_transform(df_encoded['Customer Type'])
df_encoded['Type of Travel'] = label_enc.fit_transform(df_encoded['Type of Travel'])
df_encoded['Class'] = label_enc.fit_transform(df_encoded['Class'])
df_encoded['satisfaction'] = label_enc.fit_transform(df_encoded['satisfaction'])

plt.figure(figsize=(10, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ----------------------------------------
# STEP 2: Model Training
# ----------------------------------------

# Encode original dataframe for model training
label_enc = LabelEncoder()
df['Gender'] = label_enc.fit_transform(df['Gender'])
df['Customer Type'] = label_enc.fit_transform(df['Customer Type'])
df['Type of Travel'] = label_enc.fit_transform(df['Type of Travel'])
df['Class'] = label_enc.fit_transform(df['Class'])
df['satisfaction'] = label_enc.fit_transform(df['satisfaction'])

# Split features and target
X = df.drop(['satisfaction'], axis=1)
y = df['satisfaction']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
