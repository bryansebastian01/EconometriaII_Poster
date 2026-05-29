
# %% =========================
# IMPORTACION DE LIBRERIAS
# ============================
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch
from scipy.stats import jarque_bera

# ======================================
# RUTAS DEL PROYECTO
# ======================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
RESULTS_DIR = BASE_DIR / "resultados" / "graficas"
TABLAS_DIR = BASE_DIR / "resultados" / "tablas"

# Crear carpeta resultados si no existe
RESULTS_DIR.mkdir(exist_ok=True)
TABLAS_DIR.mkdir(exist_ok=True)

# Ruta base de datos
ruta = DATA_DIR / "ipc_colombia.xls"

# Para archivos .xls (como el del proyecto) normalmente hace falta engine="xlrd".
# Nota: si tu pandas ya detecta el engine, puede funcionar sin esto, pero es más seguro indicarlo.


# ======================================
# CARGAR DATOS
# ======================================

df = pd.read_excel(ruta)

# ======================================
# LIMPIEZA DE DATOS Y FECHAS
# ======================================

meses = { # Crear columna numérica de mes a partir de la columna de texto
    'Ene': 1,
    'Feb': 2,
    'Mar': 3,
    'Abr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Ago': 8,
    'Sep': 9,
    'Oct': 10,
    'Nov': 11,
    'Dic': 12
}

df['Mes_num'] = df['Mes'].map(meses) # Crear columna de fecha a partir de año y mes numérico, asumiendo día 1

df['Fecha'] = pd.to_datetime( 
    dict(
        year=df['Año'],
        month=df['Mes_num'],
        day=1
    )
)

df = df.sort_values('Fecha')
df = df.set_index('Fecha') # Establecer la columna de fecha como índice

# ======================================
# GRAFICA SERIE ORIGINAL
# ======================================

plt.figure(figsize=(12,5))

plt.plot(df['Número Índice'])

plt.title("IPC Colombia") #se puede agregar al titulo los anos de la serie, por ejemplo "IPC Colombia (2000-2020)"
plt.xlabel("Fecha")
plt.ylabel("Número Índice")

plt.savefig(RESULTS_DIR / "serie_original.png", dpi=300)

plt.show()
#La serie original muestra una tendencia creciente a lo largo del tiempo, lo que sugiere que el IPC ha estado aumentando. Además, se pueden observar algunos picos y valles, lo que indica que hay cierta variabilidad en la serie. Sin embargo, la tendencia general es claramente ascendente.


# ======================================
# ADF SERIE ORIGINAL
# ======================================

resultado_original = adfuller(df['Número Índice'])

print("\nADF serie original:")
print("ADF Statistic:", resultado_original[0])
print("p-value:", resultado_original[1])
print("Valores críticos:", resultado_original[4]["5%"])

#CONCLUSION ADF SERIE ORIGINAL
print("La serie no es estacionaria")
print("ADF statistic:", resultado_original[0], ">", resultado_original[4]["5%"])
print("p-value:", resultado_original[1], "> 0.05")
# El resultado del ADF indica que la serie no es estacionaria, ya que el estadístico ADF es mayor que el valor crítico al 5% y el p-value es mayor que 0.05. Esto sugiere que la serie tiene una tendencia o una raíz unitaria, lo que confirma la observación visual de la gráfica original.

# ======================================
# LOGARITMO IPC
# ======================================

df['log_IPC'] = np.log(df['Número Índice'])
#Aplica logaritmo a cada uno de los indices de la columna "Numero Indice" del IPC, lo que puede ayudar a estabilizar la varianza y hacer que la serie sea más adecuada para el modelado ARIMA.

# Primera diferencia del log
df['dlog_IPC'] = df['log_IPC'].diff()
#Diferncia el logaritmo del IPC para eliminar la tendencia y hacer que la serie sea estacionaria. La función diff() calcula la diferencia entre cada valor y el valor anterior, lo que ayuda a eliminar la tendencia y estabilizar la serie.

serie_dlog = df['dlog_IPC'].dropna() 
# Eliminar el primer valor que es NaN por la diferencia  (Para el ultimo periodo no hay con que diferenciarlo por ende se genera un dato NaN, por eso se elimina con dropna())

# ======================================
# ADF LOG DIFERENCIADA
# ======================================

resultado_dlog = adfuller(serie_dlog)

print("\nADF primera diferencia log:")
print("ADF Statistic:", resultado_dlog[0])
print("p-value:", resultado_dlog[1])

#CONCLUSION ADF SERIE LOG DIFERENCIADA
print("La serie es estacionaria")
print("ADF statistic:", resultado_dlog[0], "<", resultado_dlog[4]["5%"])
print("p-value:", resultado_dlog[1], "< 0.05")
# El resultado del ADF indica que la serie es estacionaria, ya que el estadístico ADF es menor que el valor crítico al 5% y el p-value es menor que 0.05. Esto sugiere que la serie no tiene una tendencia ni una raíz unitaria, lo que confirma la observación visual de la gráfica original.

# ======================================
# GRAFICA SERIE ESTACIONARIA
# ======================================

plt.figure(figsize=(12,5))

plt.plot(serie_dlog)

plt.title("Primera diferencia del log IPC")
plt.xlabel("Fecha")
plt.ylabel("Serie estacionaria")

plt.savefig(RESULTS_DIR / "serie_estacionaria.png", dpi=300)

plt.show()

# ======================================
# FAC Y FACP
# ======================================

fig, ax = plt.subplots(1,2, figsize=(14,5))

plot_acf(serie_dlog, ax=ax[0], lags=30)
plot_pacf(serie_dlog, ax=ax[1], lags=30)

ax[0].set_title("FAC")
ax[1].set_title("FACP")

plt.savefig(RESULTS_DIR / "FAC_FACP.png", dpi=300)

plt.show()

#CONCLUSION FAC Y FACP
#1. La FAC muestra un pico significativo en el primer rezago, lo que sugiere que hay autocorrelación en el primer rezago. Esto podría indicar la presencia de un componente MA(1) en el modelo ARIMA.
#La  FAC cae lentamente a cero de forma SENOIDAL después del primer rezago, lo que sugiere que hay autocorrelación en varios rezagos, pero el primer rezago es el más significativo. Esto podría indicar un proceso autorregresivo en el modelo ARIMA.
#2. La FACP muestra un pico significativo en el primer rezago, lo que sugiere que hay autocorrelación parcial en el primer rezago. Esto podría indicar la presencia de un componente AR(1) en el modelo ARIMA.
#3. Ambos gráficos muestran que los rezagos posteriores no son significativos, lo que sugiere que un modelo ARIMA(1,1,1) podría ser adecuado para esta serie temporal, ya que captura tanto la autocorrelación como la autocorrelación parcial en el primer rezago.

#Posibilidades. Dado que en la FACP muestra un pico significativo negativo en el segundo rezago, puede haber una implicacion en que 1) la tasa en el momento t este relacionada a la tasa t-1. 2) Hay rezagos significativos en el rezago 9 al 13. Esto puede significar
# un componente estacional en el modelo, aunque no es tan claro como para afirmar que es un componente estacional de orden 12 (ARIMA(1,1,1)(0,0,0)[12]), pero si se podría considerar la posibilidad de incluir un componente estacional en el modelo ARIMA para capturar esta posible autocorrelación estacional.
# En otras palabras, un modelo ARIMA (1,1,1) puede quedarse corto.

# ======================================
# MODELOS ARIMA
# ======================================

#Nota: Para los siguientes modelos se supone d=1 por lo que la serie es diferenciada directamente por el algoritmo de estimación, por lo que no es necesario usar la serie diferenciada manualmente. 

# ======================================
# MODELO 1
# ======================================

modelo_1 = SARIMAX(
    df['log_IPC'],
    order=(1,1,0)
)

resultado_1 = modelo_1.fit()

print("\nARIMA(1,1,0)")
print(resultado_1.summary())
# CONCLUSION ARIMA (1,1,0)
# Los coeficientes son altamente significativos, sus errores no muestran autocorrelación en el primer rezago y tienen varianza estable. 
# No cumple el supuesto de normalidad en los errores y, presuntamente, los rezagos estacionales más lejanos. Al ser una variable macroeconomica
# se ve afectada por distintos shock (Paros, fenomenos naturales, crisis economicas y COVID).


# ======================================
# MODELO 2
# ======================================

modelo_2 = SARIMAX(
    df['log_IPC'],
    order=(0,1,1)
)

resultado_2 = modelo_2.fit()

print("\nARIMA(0,1,1)")
print(resultado_2.summary())
#CONCLUSION ARIMA (0,1,1)
# El Modelo 2 viola el supuesto básico de la metodología Box-Jenkins: sus residuos no son ruido blanco (Ljung-Box falló con un p-valor de 0.00). 
# Correr pronósticos con este modelo generaría estimaciones sesgadas e incorrectas.
# El AIC es mucho mayor al del modelo 1 lo que indica que el Modelo 2 es mucho peor que el Modelo 1. Esto se debe a que el modelo 2 no captura 
# la autocorrelación en el primer rezago, lo que es evidente en la FAC y FACP de la serie estacionaria.


# ======================================
# MODELO 3
# ======================================

modelo_3 = SARIMAX(
    df['log_IPC'],
    order=(1,1,1)
)

resultado_3 = modelo_3.fit()

print("\nARIMA(1,1,1)")
print(resultado_3.summary())
# CONCLUSION ARIMA (1,1,1)

# El modelo ARIMA(1,1,1) es el mejor de los tres modelos evaluados, ya que tiene el AIC más bajo (AIC: -0.28).
# No se cumple el supuesto de normalidad en los errores, pero se cumple el supuesto de no autocorrelación en los errores (Ljung-Box p-value: 0.11).
# No cumple el supuesto de normalidad en los errores y, presuntamente, los rezagos estacionales más lejanos. Al ser una variable macroeconomica
# se ve afectada por distintos shock (Paros, fenomenos naturales, crisis economicas y COVID).

# y_t = 0.8198 y_{t-1} + \epsilon_t + 0.1523 \epsilon_{t-1}
#      \underset{(0.031)}               \underset{(0.067)}

# ======================================
# COEFICIENTES Y ERRORES ESTANDAR
# ======================================

print("\nCoeficientes estimados")
print(resultado_3.params)

print("\nErrores estándar")
print(resultado_3.bse)

# ======================================
# COMPARACION AIC
# ======================================

print("\nComparación AIC")

print("ARIMA(1,1,0):", resultado_1.aic)
print("ARIMA(0,1,1):", resultado_2.aic)
print("ARIMA(1,1,1):", resultado_3.aic)
lista_aic = [resultado_1.aic, resultado_2.aic, resultado_3.aic]
menor_aic = min(lista_aic)
print("Mejor modelo:", "ARIMA(1,1,1)" if menor_aic == resultado_3.aic else "Otro modelo")

# ======================================
# RESIDUOS
# ======================================

# ORIGINAL (El primer error muestra una distorcion gigante pero esto se debe a que en el primer error no hay dato previo para diferenciarlo)
residuos = resultado_3.resid

plt.figure(figsize=(12,5))

plt.plot(residuos)

plt.title("Residuos ARIMA(1,1,1)")
plt.xlabel("Fecha")
plt.ylabel("Residuos")

plt.savefig(RESULTS_DIR / "residuos.png", dpi=300)

plt.show()

# SUGERENCIA AL ORIGINAL
# En lugar de usar todos los residuos, eliminamos los primeros valores de inicialización
residuos_limpios = resultado_3.resid.iloc[2:]  # Descarta las filas afectadas por la diferencia
plt.figure(figsize=(12,5))
plt.plot(residuos_limpios)
plt.title("Residuos Reales ARIMA(1,1,1) (Sin inicialización)")
plt.xlabel("Fecha")
plt.ylabel("Residuos")
plt.show()

# ======================================
# ACF RESIDUOS
# ======================================

plot_acf(residuos, lags=30)

plt.title("FAC Residuos")

plt.savefig(RESULTS_DIR / "FAC_residuos.png", dpi=300)

plt.show()

# ======================================
# PRUEBA LJUNG-BOX
# ======================================

ljung = acorr_ljungbox(
    residuos,
    lags=[10],
    return_df=True
)

print("\nPrueba Ljung-Box")
print(ljung)

# ======================================
# VALIDACION DE SUPUESTOS
# ======================================

# ARCH
arch_test = het_arch(
    residuos,
    nlags=5
)

# Jarque-Bera
jb = jarque_bera(residuos)

tabla_supuestos = pd.DataFrame({
    "Prueba": [
        "Ljung-Box",
        "ARCH",
        "Jarque-Bera"
    ],
    "Hipótesis nula": [
        "No autocorrelación",
        "Homocedasticidad",
        "Normalidad"
    ],
    "Estadístico": [
        ljung['lb_stat'].values[0],
        arch_test[0],
        jb.statistic
    ],
    "p-value": [
        ljung['lb_pvalue'].values[0],
        arch_test[1],
        jb.pvalue
    ]
})

print("\nValidación de supuestos")
print(tabla_supuestos)
tabla_supuestos.to_csv(
    TABLAS_DIR / "tabla_supuestos.csv",
    index=False
)
# ======================================
# PRONOSTICO
# ======================================

pronostico = resultado_3.get_forecast(steps=10)

media_log = pronostico.predicted_mean
intervalos_log = pronostico.conf_int()

# Volver a escala original IPC
media = np.exp(media_log)

intervalos = np.exp(intervalos_log)

print("\nPronóstico 10 pasos")
print(media)
# ======================================
# TABLA PRONOSTICOS
# ======================================

tabla_pronostico = pd.DataFrame({
    "Pronóstico": media,
    "Límite inferior": intervalos.iloc[:,0],
    "Límite superior": intervalos.iloc[:,1]
})

print("\nTabla pronósticos")
print(tabla_pronostico)
tabla_pronostico.to_csv(
    TABLAS_DIR / "tabla_pronosticos.csv"
)
# ======================================
# GRAFICA PRONOSTICO
# ======================================

plt.figure(figsize=(12,5))

plt.plot(
    df.index,
    df['Número Índice'],
    label='Serie observada'
)

plt.plot(
    media.index,
    media,
    label='Pronóstico'
)

plt.fill_between(
    intervalos.index,
    intervalos.iloc[:,0],
    intervalos.iloc[:,1],
    alpha=0.3
)

plt.legend()

plt.title("Pronóstico ARIMA(1,1,1)")

plt.savefig(RESULTS_DIR / "pronostico.png", dpi=300)

plt.show()
# ======================================
# FAC RESIDUALES AL CUADRADO
# ======================================

residuos2 = residuos**2

plot_acf(residuos2, lags=30)
plt.title("FAC Residuales al Cuadrado")
plt.savefig(
    RESULTS_DIR / "FAC_residuos_cuadrado.png",
    dpi=300
)
plt.show()
# %%
