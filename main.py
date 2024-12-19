from preprocesamiento import cargar_datos, limpiar_datos
from analisis import realizar_analisis
from modelado import entrenar_modelos, evaluar_modelos

def main():
    # Cargar y limpiar los datos
    df = cargar_datos("EmployeesData.csv")
    df = limpiar_datos(df)
    print ('primera parte ok')

    # Análisis exploratorio
    realizar_analisis(df)
    print ('segunda parte ok')

    # Modelado y evaluación
    X_train, X_test, y_train, y_test, rf, rf_balanced = entrenar_modelos(df)
    evaluar_modelos(X_test, y_test, rf, rf_balanced)
    print ('tercera parte ok')

if __name__ == "__main__":
    main()
