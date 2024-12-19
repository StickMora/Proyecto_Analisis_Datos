# Proyecto_Analisis_Datos

# Análisis y Predicción de Datos de Empleados

Este proyecto consiste en analizar un dataset de empleados para realizar predicciones sobre si un empleado tomará licencia (LeaveOrNot) utilizando modelos de Machine Learning, específicamente Random Forest.

## Dependencias

El proyecto requiere las siguientes bibliotecas de Python:

- pandas
- numpy
- matplotlib
- scikit-learn
- seaborn (opcional para gráficos)

Puedes instalar estas dependencias utilizando pip:

pip install pandas numpy matplotlib scikit-learn seaborn


## Ejecución del Proyecto

1. **Cargar el Dataset:** Se carga el dataset desde un archivo CSV.
2. **Preprocesamiento de los Datos:** El código maneja valores faltantes, convierte variables categóricas y elimina valores atípicos.
3. **Análisis Exploratorio:** Se realizan varias visualizaciones para entender la distribución de los datos.
4. **Modelado de Datos:** Se entrenan modelos RandomForest y se calculan métricas de desempeño.

## Estructura del Proyecto

- **preprocesamiento.py:** Carga y limpieza de datos.
- **analisis.py:** Análisis exploratorio y visualización de datos.
- **modelado.py:** Entrenamiento de modelos y evaluación de métricas.

## Cómo Ejecutarlo

1. Clona este repositorio.
2. Ejecuta el archivo `main.py`
3. Revisa las métricas generadas para evaluar el desempeño de los modelos.

## Resultados

- Comparación entre modelos con y sin ajuste de clases (class_weight='balanced').
- Análisis de la propensión de los empleados a tomar licencias según su edad.
- Evaluación del balance de clases en el dataset.