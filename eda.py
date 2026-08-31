import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carga de datos
print("========== Carga de Datos ==========")
data = pd.read_csv('./data.csv')
print(data.shape)

# Eliminacion de datos duplicados
print("========= Duplicados =========")
print(data.duplicated().sum())
print("/=/=/ Limpiando datos /=/=/")
data = data.drop_duplicates()
print(data.shape)

# Exploracion basica de los datos
print("========== Info del DataSet =========")
data.info()
print("/=/=/ Columnas del DataSet /=/=/")
print(data.columns)
print("/=/=/ Head del DataSet /=/=/")
print(data.head())

# Revision de datos faltantes
print("========== Datos faltantes ==========")
print("Null data:\n", data.isnull().sum())
print("Null data: ", data.isnull().sum().sum())
print("/=/=/=/=/=/=/=/=/=/=")
print("? Data:\n", (data == '?').sum())
print("? Data total: ", (data == '?').sum().sum())
print("/=/=/ Transformando ? a valores na /=/=/")
data = data.replace('?', np.nan)
print("Null data nuevo: ", data.isnull().sum().sum())

# Conteo de datos por columna
print("========== Conteo de Datos por columna ==========")
cols = data.columns
for i in cols:
    print(data[i].value_counts())
    print("/=/=/=/=/=/=/=/=/=/=")

'''
Columnas innecesarias
id: Valores unicos, no aporta nada
Profession: Casi todos los datos son estudiantes, menos del 1% no. No aporta nada
Work Pressure: Solo hay 3 valores distintos a 0, no aporta valor
Job Satisfaction: Solo hay 8 datoos que no son 0, no aporta valor

Se harán graficos sobre las demas columnas en relación a la columna target para ver
su relación.
'''

# Plot sobre la ciudades y grados academicos
fig, axes = plt.subplots(1, 2, figsize=(16,6))


sns.countplot(data=data, y="City", hue="Depression", order=data["City"].value_counts().index, ax=axes[0])
sns.countplot(data=data, y="Degree", hue="Depression", order=data["Degree"].value_counts().index, ax=axes[1])

axes[0].set_title("Distirbución de ciudades según depresión")
axes[0].set_xlabel("Cantidad")
axes[0].set_ylabel("Ciudad")

axes[1].set_title("Distirbución de Grado academico/Carrera según depresión")
axes[1].set_xlabel("Cantidad")
axes[1].set_ylabel("Grado academico/Carrera")

#plt.tight_layout()
#plt.show()

# Distribuciones de edad
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.boxplot(data=data, x="Depression", y="Age", ax=axes[0])
sns.boxplot(data=data, y="Age", ax=axes[1])

axes[0].set_title("Distribución de edad según depresión")
axes[0].set_xlabel("Depression")
axes[0].set_ylabel("Edad")

axes[1].set_title("Distribución de edad")
axes[1].set_xlabel("")
axes[1].set_ylabel("Edad")

#plt.tight_layout()
#plt.show()

# Pression y satisfacción academica
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.countplot(data=data, x="Academic Pressure", hue="Depression", ax=axes[0])
sns.countplot(data=data, x="Study Satisfaction", hue= "Depression", ax=axes[1])

axes[0].set_title("Distribución de presión academica según depresión")
axes[0].set_xlabel("Presión Academica")
axes[0].set_ylabel("Cantidad de personas")

axes[1].set_title("Distribución de satisfacción de estudio según depresión")
axes[1].set_xlabel("Satisfaccion de estudio")
axes[1].set_ylabel("Cantidad de personas")

#plt.tight_layout()
#plt.show()

# Familiares con registros de enfermedades mentales y pensamientos suicidas
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.countplot(data=data, x="Have you ever had suicidal thoughts ?", hue="Depression", ax=axes[0])
sns.countplot(data=data, x="Family History of Mental Illness", hue= "Depression", ax=axes[1])

axes[0].set_title("Distribución de pensamientos suicidas según depresión")
axes[0].set_xlabel("Pensamientos suicidas")
axes[0].set_ylabel("Cantidad de personas")

axes[1].set_title("Distribución de familiares con probelmas mentales según depresión")
axes[1].set_xlabel("Familiares con probelmas mentales")
axes[1].set_ylabel("Cantidad de personas")

#plt.tight_layout()
#plt.show()

# Plot de duración de sueño y horas de estudio
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.countplot(data=data, x="Sleep Duration", hue="Depression", ax=axes[0])
sns.countplot(data=data, x="Work/Study Hours", hue= "Depression", ax=axes[1])

axes[0].set_title("Distribución de horas de sueño según depresión")
axes[0].set_xlabel("Horas de sueño")
axes[0].set_ylabel("Cantidad de personas")

axes[1].set_title("Distribución de horas de estudio/trabajo según depresión")
axes[1].set_xlabel("Horas de estudio/trabajo")
axes[1].set_ylabel("Cantidad de personas")

#plt.tight_layout()
#plt.show()

# Plot de genero conn estres financiero
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.countplot(data=data, x="Gender", hue="Depression", ax=axes[0])
sns.countplot(data=data, x="Financial Stress", hue= "Depression", ax=axes[1])

axes[0].set_title("Distribución de género según depresión")
axes[0].set_xlabel("Género")
axes[0].set_ylabel("Cantidad de personas")

axes[1].set_title("Distribución de estrés financiero según depresión")
axes[1].set_xlabel("Estrés financiero")
axes[1].set_ylabel("Cantidad de personas")

#plt.tight_layout()
#plt.show()

#Plot de CGPA (Promedios) y Habitos alimenticios
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.histplot(data=data, x="CGPA", hue="Depression", multiple="dodge", bins=10, ax=axes[0])
sns.countplot(data=data, x="Dietary Habits", hue= "Depression", ax=axes[1])

axes[0].set_title("Distribución de promedio académico según depresión")
axes[0].set_xlabel("Promedio académico")
axes[0].set_ylabel("Frecuencia")

axes[1].set_title("Distribución de hábitos alimenticios según depresión")
axes[1].set_xlabel("Hábitos alimenticios")
axes[1].set_ylabel("Cantidad de personas")

#plt.tight_layout()
#plt.show()

'''
Las  columnas de City y Degree pueden estar inclinadas a solo un tipo de muestra,
es decir que los datos fueron mayormennnte recabados en esas ciudades y a alumnos
por lo que no genera relevancia a la prediccion si apuntaramos personas al rededor
del mundo.

Las columnas restantes guardan unan relacion enntre sus datos y el target,
principalmente Presion academica, promedio, habitos alimenticios, horas de estudio,
estres financiero y pensamientos suicidas.
'''

# Limpieza de datos
clean_data = data.copy()

# Borrando las columnas innecesarias
clean_data = clean_data.drop(columns=["id", "City", "Profession", "Work Pressure", "Job Satisfaction", "Degree"])

# Transformacion de datos a datos binarios
clean_data["Have you ever had suicidal thoughts ?"] = clean_data["Have you ever had suicidal thoughts ?"].map(
    {
        "Yes": 1,
        "No": 0
    }
)
clean_data["Family History of Mental Illness"] = clean_data["Family History of Mental Illness"].map(
    {
        "Yes": 1,
        "No": 0
    }
)
clean_data["Gender"] = clean_data["Gender"].map(
    {
        "Male": 1,
        "Female": 0
    }
)

# Transofrmacion de clase a escala numerica
clean_data["Sleep Duration"] = clean_data["Sleep Duration"].map(
    {
        "Less than 5 hours": 0,
        "5-6 hours": 1,
        "7-8 hours": 2,
        "More than 8 hours": 3
    }
)
clean_data["Dietary Habits"] = clean_data["Dietary Habits"].map(
    {
        "Unhealthy": 0,
        "Moderate": 1,
        "Healthy": 2
    }
)

# Borrando edades mayores a 30, representan muy pocos datos
clean_data = clean_data[clean_data["Age"] <= 30]

clean_data = clean_data.rename(columns={
    "Have you ever had suicidal thoughts ?": "Suicidal Thoughts",
    "Family History of Mental Illness": "Family Illness"
})

# Exploracion basica de los datos
print("========== Info del DataSet =========")
clean_data.info()
print("/=/=/ Columnas del DataSet /=/=/")
print(clean_data.columns)
print("/=/=/ Head del DataSet /=/=/")
print(clean_data.head())
print("/=/=/ Forma del DataSet /=/=/")
print(clean_data.shape)

# Revision de datos faltantes
print("========== Datos faltantes ==========")
print("Null data:\n", clean_data.isnull().sum())
print("Null data: ", clean_data.isnull().sum().sum())
print("/=/=/=/=/=/=/=/=/=/=")
print("? Data:\n", (clean_data == '?').sum())
print("? Data total: ", (clean_data == '?').sum().sum())
print("/=/=/ Transformando ? a valores na /=/=/")
clean_data = clean_data.replace('?', np.nan)
print("Null data nuevo: ", clean_data.isnull().sum().sum())
print("/=/=/ Borrando na del DataSet /=/=/")
clean_data = clean_data.dropna()
print("Null data nuevo: ", clean_data.isnull().sum().sum())

print("========== Forma final ==========")
clean_data.info()
print(clean_data.shape)

# clean_data.to_csv("clean_data.csv", index=False)