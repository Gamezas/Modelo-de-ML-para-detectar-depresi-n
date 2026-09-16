import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Carga de los datos
data = pd.read_csv("./clean_data.csv")

# Separación de la columna target
X = data.drop(columns=["Depression"])
y = data["Depression"]

# Slit de train, val y test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=67)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=67)

# Configuraciones distintas para encontrar la forma óptima
configs = [
    {"n_estimators": 50,  "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 8},
    {"n_estimators": 200, "max_depth": 12},
    {"n_estimators": 300, "max_depth": None} 
]

results = []
models = []

for cfg in configs:
    model = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        bootstrap=True,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=12
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    train_loss = log_loss(y_train, model.predict_proba(X_train))
    val_loss = log_loss(y_val, model.predict_proba(X_val))

    gap = train_acc - val_acc

    results.append({
        "n_estimators":cfg["n_estimators"],
        "max_depth": cfg["max_depth"],
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "gap": gap
    })
    models.append(model)

    print(f"n_estimators={cfg['n_estimators']:<4} max_depth={str(cfg['max_depth']):<5} "
          f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
          f"train_loss={train_loss:.3f} val_loss={val_loss:.3f} gap={gap:.3f}")

# Transformación de los resultados a un dataframe para explorarlo más fácil con pandas
results_df = pd.DataFrame(results)
best_idx = results_df["val_acc"].idxmax()
best_model = models[best_idx]

print("\nMejor configuración:")
print(results_df.loc[best_idx])

y_pred_test = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"\nAccuracy en test: {test_accuracy * 100:.2f}%")

# Matriz de confusión del mejor modelo
cm = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Matriz de Confusión - Test")
plt.tight_layout()
plt.show()

# Graficación de las configuraciones tran vs val
labels = [f"{c['n_estimators']}/{c['max_depth']}" for c in configs]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(labels, results_df["train_loss"], marker="o", label="Train")
ax.plot(labels, results_df["val_loss"], marker="o", label="Val")
ax.set_title("Log Loss: Train vs Val")
ax.set_xlabel("n_estimators / max_depth")
ax.set_ylabel("Log Loss")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()