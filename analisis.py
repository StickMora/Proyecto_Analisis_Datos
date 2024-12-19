import matplotlib.pyplot as plt

def realizar_analisis(df):
    """Realiza el análisis exploratorio de datos con visualizaciones."""
    
    # Graficar la distribución de los sexos con un gráfico de torta
    df['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(6,6))
    plt.title("Distribución de Géneros")
    plt.ylabel('')
    plt.show()

    # Distribución de niveles de estudio con subplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    df['Education'].value_counts().plot(kind='bar', ax=ax[0])
    ax[0].set_title("Distribución de Niveles de Estudio")
    ax[0].set_xlabel("Nivel de Estudio")
    ax[0].set_ylabel("Frecuencia")
    df['Education'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax[1])
    ax[1].set_title("Distribución de Niveles de Estudio (Torta)")
    plt.show()

    # Graficar la distribución de la edad vs. LeaveOrNot
    df[df['LeaveOrNot'] == 'Leave']['Age'].plot(kind='hist', alpha=0.5, label='Leave', bins=20)
    df[df['LeaveOrNot'] == 'Not Leave']['Age'].plot(kind='hist', alpha=0.5, label='Not Leave', bins=20)
    plt.title("¿Son los jóvenes más propensos a tomar licencias?")
    plt.xlabel("Edad")
    plt.legend()
    plt.show()

    # Verificar balance de clases (LeaveOrNot)
    df['LeaveOrNot'].value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title("Distribución de la Clase 'LeaveOrNot'")
    plt.xlabel("Clase")
    plt.ylabel("Frecuencia")
    plt.show()
