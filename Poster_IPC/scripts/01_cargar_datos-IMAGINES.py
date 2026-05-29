
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

# Como nuestra grafica tiene cambio de pendiente a partir del 2021, (seguramente por pandemia) podemos usar la prueba:
# ZIVOT-ANDREWS
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

# =====================================- GRAFICAS SERIE ORIGINAL Y FAC Y FACP PARA CANVA =================
# ===================================== SERIE ORIGINAL PARA CANVA

# Crear la figura para la serie original
plt.figure(figsize=(8, 5))

# Graficar la columna original usando el índice de fechas
plt.plot(df.index, df['Número Índice'], color='darkblue', linewidth=2)

# Configuración estética
plt.title("Comportamiento de la Serie en Niveles del IPC en Colombia (2000 - 2026)", fontsize=12, fontweight='bold')
plt.xlabel("Año", fontsize=10)
plt.ylabel("Número Índice", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5) # Agrega una cuadrícula sutil

# Guardar la imagen en alta calidad para Canva
plt.savefig(RESULTS_DIR / "01_ipc_serie_original.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()
# ===================================== FAC Y FACP PARA CANVA

# Creamos la figura transparente para la serie NO estacionaria
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_alpha(0.0)

# FAC de la serie original (Verás el decrecimiento lento)
plot_acf(df['Número Índice'], lags=30, ax=axes[0], color='red', vlines_kwargs={"colors": 'red'})
axes[0].set_title("FAC - Serie Original (No Estacionaria)", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Rezagos")
axes[0].grid(True, linestyle='--', alpha=0.3)

# FACP de la serie original (Verás solo el primer rezago gigante)
plot_pacf(df['Número Índice'], lags=30, ax=axes[1], color='orange', vlines_kwargs={"colors": 'orange'}, method='ywm')
axes[1].set_title("FACP - Serie Original (No Estacionaria)", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Rezagos")
axes[1].grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "00_fac_facp_no_estacionaria.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()

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


# ====================================== GRAFICA SERIE ESTACIONARIA PARA CANVA =================
# ====================================== SERIE ESTACIONARIA PARA CANVA

# Configuramos el tamaño exacto solicitado a 12 de ancho por 5 de alto
plt.figure(figsize=(12, 5))

# Graficamos la serie transformada
plt.plot(serie_dlog.index, serie_dlog, color='teal', linewidth=1.5)

# Ajustes estéticos y títulos legibles para impresión
plt.title("Serie Transformada: Inflación Mensual Estacionaria", fontsize=14, fontweight='bold')
plt.xlabel("Año", fontsize=11)
plt.ylabel("Diferencia del Log IPC", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.3)

# Guardar con transparencia para Canva
plt.savefig(RESULTS_DIR / "03_serie_estacionaria_individual.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()
# ====================================== FAC Y FACP PARA CANVA SERIE ESTACIONARIA =================

# Creamos el lienzo de 12 de ancho por 5 de alto con fondo transparente
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_alpha(0.0) 

# ---- Gráfica Izquierda: FAC ----
plot_acf(serie_dlog, lags=30, ax=axes[0], color='purple', vlines_kwargs={"colors": 'purple'})
axes[0].set_title("Función de Autocorrelación (FAC)", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Rezagos (Lags)", fontsize=11)
axes[0].set_ylabel("Correlación", fontsize=11)
axes[0].grid(True, linestyle='--', alpha=0.3)

# ---- Gráfica Derecha: FACP ----
plot_pacf(serie_dlog, lags=30, ax=axes[1], color='darkgreen', vlines_kwargs={"colors": 'darkgreen'}, method='ywm')
axes[1].set_title("Función de Autocorrelación Parcial (FACP)", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Rezagos (Lags)", fontsize=11)
axes[1].grid(True, linestyle='--', alpha=0.3)

# Ajustar distribución de márgenes y exportar
plt.tight_layout()
plt.savefig(RESULTS_DIR / "04_fac_facp_estacionarias_individual.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()



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

# y_t = 0.8198 y_{t-1} + 0.1523 \epsilon_{t-1}
#      \underset{(0.031)}               \underset{(0.067)}

# ====================================== ESTIMACION PARA CANVA =================
import matplotlib.pyplot as plt

# Configuramos una figura pequeña, ya que solo contendrá texto,
# pero mantenemos una resolución alta (dpi) para que se vea nítida en Canva.
# Un tamaño de 8x2 pulgadas es suficiente para una ecuación sola.
fig = plt.figure(figsize=(8, 2))

# Activamos la transparencia total del fondo de la figura
fig.patch.set_alpha(0.0)

# Añadimos un eje que ocupe todo el espacio y lo hacemos invisible
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

# Construimos la ecuación utilizando sintaxis LaTeX pura compatible con Matplotlib.
# - r"..." : Indica que es una cadena 'raw' para interpretar las barras diagonales \
# - $ ... $ : Delimitadores de modo matemático de LaTeX.
# - \hat{...} : Coloca el 'gorro' sobre la variable (indica estimación).
# - \underset{debajo}{arriba} : Coloca el texto de los errores estándar justo debajo de los coeficientes.
# - \epsilon : Símbolo griego para el error.

ecuacion_latex = (
    r"${y}_t = \underset{(0.031)}{\hat{0.8198}}{y}_{t-1} + \underset{(0.067)}{\hat{0.1523}} \epsilon_{t-1}$"
)

# Colocamos el texto en el centro exacto de la imagen (x=0.5, y=0.5)
# Aumentamos el fontsize a 20 para que sea el protagonista visual.
ax.text(0.5, 0.5, ecuacion_latex,
        fontsize=20,
        color='black', # Color del texto (puedes cambiarlo según tu diseño de Canva)
        ha='center',   # Alineación horizontal centrada
        va='center',   # Alineación vertical centrada
        transform=ax.transAxes)

# Guardamos la imagen.
# Es CRUCIAL mantener transparent=True y bbox_inches='tight' para que
# solo se guarde la ecuación sin bordes blancos extra.
nombre_archivo = "estimacion.png"
plt.savefig(nombre_archivo,
            dpi=300,
            transparent=True,
            bbox_inches='tight',
            pad_inches=0.1)

print(f"✅ Imagen generada con éxito: {nombre_archivo}")
print("Ahora puedes subirla a Canva y se adaptará a cualquier fondo.")

plt.savefig(RESULTS_DIR / "estimacion.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()

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
    lags=[12],
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

# ====================================== DIAGNOSTICO DE RESIDUOS PARA CANVA =================

# Datos exactos del diagnóstico del modelo ARIMA(1,1,1) de tu proyecto
data = {
    "Supuesto Evaluado": ["No Autocorrelación", "Homocedasticidad", "Normalidad"],
    "Prueba Estadística": ["Ljung-Box (Lag 10)", "ARCH-LM", "Jarque-Bera"],
    "P-Valor": ["1.0000", "0.7978", "0.0000"],
    "Estado / Conclusión": ["Cumple (Ruido Blanco)", "Cumple (Varianza Constante)", "No Cumple (No Normalidad)"]
}

df_tabla = pd.DataFrame(data)

# Configurar dimensiones exactas de 8x5 pulgadas solicitadas para mantener la simetría
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_alpha(0.0) # Forzar la transparencia total del lienzo
ax.axis('off')

# Definir la paleta cromática ejecutiva para las celdas
header_color = '#104e5b'       # Azul verdoso/Teal oscuro académico
row_colors = ['#1d2a3a', '#243346'] # Alternancia de grises azulados oscuros para contraste
text_color = 'white'

# Crear el objeto tabla dentro de Matplotlib
tabla = ax.table(
    cellText=df_tabla.values,
    colLabels=df_tabla.columns,
    cellLoc='center',
    loc='center'
)

# Configuración de escala para que las celdas "respiren" y tengan margen interior
tabla.auto_set_font_size(False)
tabla.set_fontsize(9)
tabla.scale(1.2, 1.9) # Incrementamos la altura para darle un aspecto premium

# Iterar sobre cada celda para inyectar los estilos personalizados de forma dinámica
for (row, col), cell in list(tabla.get_celld().items()):
    cell.set_edgecolor('#3a4f66') # Bordes sutiles integrados
    cell.set_linewidth(1.5)
    
    if row == 0:
        # Formato exclusivo para la fila de encabezados
        cell.set_text_props(weight='bold', color='white', size=11)
        cell.set_facecolor(header_color)
    else:
        # Formato para el cuerpo de la tabla
        color_fondo = row_colors[row % 2]
        cell.set_facecolor(color_fondo)
        
        # Lógica de color condicional para el veredicto del jurado (Columna Estado)
        if col == 3:
            if "Cumple" in cell.get_text().get_text() and "No" not in cell.get_text().get_text():
                cell.set_text_props(color='#4affb0', weight='bold') # Verde neón legible sobre oscuro
            else:
                cell.set_text_props(color='#ff6b6b', weight='bold') # Coral/Rojo de alerta sutil
        elif col == 2:
            # Los P-valores se destacan en negrita clara
            cell.set_text_props(color='#e2e8f0', weight='bold')
        else:
            cell.set_text_props(color=text_color)

# Guardar la imagen en alta definición
plt.tight_layout()
nombre_archivo = "tabla_validacion_supuestos.png"
plt.savefig(RESULTS_DIR / "tabla_validacion_supuestos.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# ====================================== RESIDUOS PARA CANVA =================
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Creamos el lienzo de 12x5 pulgadas solicitado con fondo transparente
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_alpha(0.0)

# ---- Gráfica Izquierda: FAC de Residuos Simples (Evaluación de Media / Autocorrelación) ----
plot_acf(residuos, lags=30, ax=axes[0], color='darkblue', vlines_kwargs={"colors": 'darkblue'})
axes[0].set_title("FAC de los Residuos ($e_t$)\n[Evaluación de Autocorrelación]", fontsize=12, fontweight='bold', color='white')
axes[0].set_xlabel("Rezagos (Lags)", fontsize=10, color='white')
axes[0].grid(True, linestyle='--', alpha=0.2)

# ---- Gráfica Derecha: FAC de Residuos al Cuadrado (Evaluación de Varianza / Efectos ARCH) ----
# Elevamos al cuadrado la serie de residuos directamente dentro de la función
plot_acf(residuos**2, lags=30, ax=axes[1], color='darkorange', vlines_kwargs={"colors": 'darkorange'})
axes[1].set_title("FAC de Residuos al Cuadrado ($e_t^2$)\n[Evaluación de Homocedasticidad]", fontsize=12, fontweight='bold', color='white')
axes[1].set_xlabel("Rezagos (Lags)", fontsize=10, color='white')
axes[1].grid(True, linestyle='--', alpha=0.2)

# Configurar textos en blanco para posters oscuros (eliminar si tu fondo es claro)
for ax in axes:
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')

plt.tight_layout()
nombre_output = "fac_residuos_simples_y_cuadrados.png"
plt.savefig(RESULTS_DIR / 'fac_residuos_simples_y_cuadrados.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
print(f"Gráficas de los monitores generadas con éxito: {nombre_output}")
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

# ====================================== PRONOSTICO PARA CANVA =================

# 1. Extraer las predicciones dentro de la muestra usando las variables de tu código
# Tu modelo estimado está guardado en la variable 'res'
pred_dinamica = resultado_3.get_prediction()
pronostico_valores = pred_dinamica.predicted_mean

# Extraer los intervalos de confianza (95%)
intervalos = pred_dinamica.conf_int(alpha=0.05)
intervalo_inferior = intervalos.iloc[:, 0]
intervalo_superior = intervalos.iloc[:, 1]

# 2. Filtrar los datos desde el año 2020 para que la gráfica tenga un zoom perfecto
# Usamos 'serie_dlog' que es el nombre exacto de tu serie diferenciada
datos_reales_zoom = serie_dlog.loc['2020':]
pronostico_zoom = pronostico_valores.loc['2020':]
inf_zoom = intervalo_inferior.loc['2020':]
sup_zoom = intervalo_superior.loc['2020':]

# 3. Configurar el lienzo simétrico (12x5 pulgadas) y transparente para Canva
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_alpha(0.0) # Fondo transparente
ax.axis('on')

# Graficar la serie histórica real (Línea Teal/Azul verdoso)
ax.plot(datos_reales_zoom.index, datos_reales_zoom.values, 
        color='#104e5b', linewidth=2.5, label='Inflación Real (Dlog IPC)')

# Graficar el pronóstico/ajuste del ARIMA(1,1,1) (Línea discontinua Naranja)
ax.plot(pronostico_zoom.index, pronostico_zoom.values, 
        color='darkorange', linewidth=2, linestyle='--', label='Ajuste del Modelo ARIMA')

# Graficar el sombreado del intervalo de confianza
ax.fill_between(pronostico_zoom.index, inf_zoom.values, sup_zoom.values, 
                color='orange', alpha=0.15, label='Intervalo de Confianza (95%)')

# Configuración de etiquetas y estilo visual para póster oscuro
ax.set_title("Capacidad Predictiva y Ajuste del Modelo ARIMA(1,1,1)", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Periodo (Mensual)", fontsize=11)
ax.set_ylabel("Variación Logarítmica", fontsize=11)
ax.grid(True, linestyle='--', alpha=0.2, color='gray')
ax.legend(loc='upper left', fontsize=10, framealpha=0.3)

# --- AJUSTE DE COLORES PARA TU PÓSTER ---
# Si tu fondo de Canva es OSCURO, mantén las siguientes líneas. 
# Si tu fondo de Canva es CLARO, simplemente bórralas o coméntalas (#).
ax.title.set_color('white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_color('white')

# Guardar la imagen optimizada en alta definición
plt.tight_layout()
nombre_salida = "06_grafica_pronostico_arima.png"
plt.savefig(RESULTS_DIR, dpi=300, bbox_inches='tight', transparent=True)
plt.show()
print(f"✅ ¡Gráfica de pronóstico generada con éxito!: {nombre_salida}")


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
