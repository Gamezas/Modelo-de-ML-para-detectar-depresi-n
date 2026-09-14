import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Cargamos los datos
data = pd.read_csv("./clean_data.csv")

# Separamos la columna target
X = data.drop(columns=["Depression"])
y = data["Depression"]

# Separamos train, val y test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=67)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=67)

# Carga del modelo
model = RandomForestClassifier(n_estimators=20, random_state=12)

# Entrenar el modelo
model.fit(X_train, y_train)

# Predicción del modelo
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
