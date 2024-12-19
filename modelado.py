import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def entrenar_modelos(df):
    """Entrena los modelos de Random Forest y devuelve las particiones de los datos."""
    # Preparar los datos para el modelado
    X = df.drop(columns=['LeaveOrNot'])
    y = df['LeaveOrNot']

    # Convertir variables categóricas en variables dummies
    X = pd.get_dummies(X, drop_first=True)

    # Dividir el dataset en entrenamiento y prueba (estratificado)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Entrenar un Random Forest sin class_weight
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # Entrenar un Random Forest con class_weight="balanced"
    rf_balanced = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_balanced.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, rf, rf_balanced

def evaluar_modelos(X_test, y_test, rf, rf_balanced):
    """Evalúa los modelos entrenados y muestra las métricas de rendimiento."""
    # Hacer predicciones
    y_pred = rf.predict(X_test)
    y_pred_balanced = rf_balanced.predict(X_test)

    # Calcular accuracy y f1 score
    accuracy_rf = accuracy_score(y_test, y_pred)
    accuracy_rf_balanced = accuracy_score(y_test, y_pred_balanced)
    f1_rf = f1_score(y_test, y_pred, pos_label='Leave')
    f1_rf_balanced = f1_score(y_test, y_pred_balanced, pos_label='Leave')

    # Mostrar métricas
    print(f"Accuracy (Random Forest): {accuracy_rf}")
    print(f"Accuracy (Balanced Random Forest): {accuracy_rf_balanced}")
    print(f"F1 Score (Random Forest): {f1_rf}")
    print(f"F1 Score (Balanced Random Forest): {f1_rf_balanced}")

    # Matrices de confusión
    cm_rf = confusion_matrix(y_test, y_pred, labels=['Leave', 'Not Leave'])
    cm_rf_balanced = confusion_matrix(y_test, y_pred_balanced, labels=['Leave', 'Not Leave'])

    # Mostrar matrices de confusión
    ConfusionMatrixDisplay(cm_rf, display_labels=['Leave', 'Not Leave']).plot()
    ConfusionMatrixDisplay(cm_rf_balanced, display_labels=['Leave', 'Not Leave']).plot()
    plt.show()
