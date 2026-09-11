import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carga de datos
print("========== Carga de Datos ==========")
data = pd.read_csv('./clean_data.csv')
print(data.shape)

columns_to_encode = [
    "Academic Pressure",
    "Study Satisfaction",
    "Financial Stress",
    "Sleep Duration",
    "Dietary Habits"
]

clean_data = pd.get_dummies(
    data,
    columns=columns_to_encode,
    prefix=columns_to_encode,
    dtype=int
)

print("========== DATASET V2 ==========")

clean_data.info()

print("\nColumnas:")
print(clean_data.columns.tolist())

print("\nPrimeras filas:")
print(clean_data.head())

print("\nForma:")
print(clean_data.shape)

clean_data.to_csv("clean_datav2.csv", index=False)