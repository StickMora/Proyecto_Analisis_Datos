import pandas as pd

def cargar_datos(file_path: str) -> pd.DataFrame:
    """Carga el dataset desde un archivo CSV."""
    df = pd.read_csv(file_path)
    return df

def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza la limpieza y preprocesamiento de los datos."""
    # Convertir la columna 'LeaveOrNot' a etiquetas categóricas
    df['LeaveOrNot'] = df['LeaveOrNot'].replace({1: 'Leave', 0: 'Not Leave'})

    # Eliminar filas con valores faltantes en 'ExperienceInCurrentDomain' y 'JoiningYear'
    df.dropna(subset=['ExperienceInCurrentDomain', 'JoiningYear'], inplace=True)

    # Imputar valores faltantes en 'Age' con la media
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')  # Asegurar que 'Age' sea numérico
    df['Age'].fillna(df['Age'].mean(), inplace=True)

    # Imputar valores faltantes en 'PaymentTier' con la moda
    df['PaymentTier'].fillna(df['PaymentTier'].mode()[0], inplace=True)

    # Seleccionar solo las columnas numéricas para calcular el IQR
    numeric_cols = df.select_dtypes(include=['number']).columns

    # Calcular IQR para las columnas numéricas
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    # Filtrar registros fuera del rango permitido por el IQR
    df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

    return df
