import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset

# Configuración inicial
plt.style.use('ggplot')
pd.options.display.float_format = '{:,.2f}'.format

# Paso 1: Carga y conversión de fechas
data = pd.read_csv("Pronosticos\Cuantitativos\Ventas.csv")

# Generar fechas asumiendo orden cronológico y año 2023
data['Fecha'] = pd.to_datetime(
    '2023-' + (data.index + 1).astype(str) + '-01',
    format='%Y-%m-%d'
)
data = data.set_index('Fecha').asfreq('MS')
data.index = data.index.to_period('M').to_timestamp()  # Forzar formato datetime

# Paso 2: Cálculo de pronósticos (sin cambios)
# Promedio móvil
rolling_mean = data['Ventas'].rolling(window=3).mean()

# Suavización exponencial
model_exp = ExponentialSmoothing(data['Ventas'], trend='additive').fit(smoothing_level=0.3)
forecast_exp = model_exp.forecast(steps=3)

# ARIMA
model_arima = ARIMA(data['Ventas'], order=(1, 1, 1)).fit()
forecast_arima = model_arima.forecast(steps=3)

# Método de pesos aplicados
weights = np.array([3, 2, 1])
weighted_avg = (data['Ventas'].rolling(window=3)
                              .apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)
forecast_weighted = [weighted_avg.iloc[-1]] * 3)  # Usar iloc para acceso seguro

# Generación de fechas futuras
last_date = data.index[-1]
forecast_dates = pd.date_range(
    start=last_date + DateOffset(months=1),
    periods=3,
    freq='MS'
)

# Resto del código manteniendo tablas y gráficos...
# [Mantener igual desde la creación de tablas hasta el final]

# Paso 3: Creación de tablas de resultados
historical_table = pd.DataFrame({
    'Ventas Reales': data['Ventas'],
    'Media Móvil (3M)': rolling_mean,
    'Pesos Aplicados': weighted_avg
}).tail(6)

forecast_table = pd.DataFrame({
    'Suavización Exponencial': forecast_exp.values,
    'ARIMA': forecast_arima.values,
    'Pesos Aplicados': forecast_weighted
}, index=forecast_dates)

print("\n➤ Tabla de Datos Históricos y Métricas:")
print(historical_table.to_markdown(tablefmt="grid", stralign='center', numalign='center'))

print("\n➤ Tabla de Pronósticos para el Próximo Trimestre:")
print(forecast_table.to_markdown(tablefmt="grid", stralign='center', numalign='center'))

# Paso 4: Visualización (sin cambios)
plt.figure(figsize=(14, 7))
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b-%Y"))

plt.plot(data.index, data['Ventas'], 'o-', color='#2c3e50', label='Ventas Históricas', alpha=0.9)
plt.plot(rolling_mean.index, rolling_mean, '--', color='#e74c3c', label='Media Móvil (3M)')
plt.plot(weighted_avg.index, weighted_avg, ':', color='#27ae60', linewidth=2, label='Pesos Aplicados')

colors = ['#9b59b6', '#f1c40f', '#e67e22']
methods = [forecast_exp, forecast_arima, pd.Series(forecast_weighted, index=forecast_dates)]
for color, method, label in zip(colors, methods, ['Suavización', 'ARIMA', 'Pesos']):
    plt.plot(method.index, method.values, 'X--', color=color, markersize=10, label=label)

plt.title('Análisis Predictivo de Ventas', pad=20, fontsize=16)
plt.xlabel('Período', labelpad=15)
plt.ylabel('Unidades', labelpad=15)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()