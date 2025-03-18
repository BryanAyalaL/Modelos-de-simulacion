import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, chisquare

# Configuración de parámetros
lambda_param = 5
muestra = 1200

# Generar datos de Poisson
np.random.seed(42)  # Para reproducibilidad
datos = np.random.poisson(lam=lambda_param, size=muestra)

# Crear diagrama de barras
valores_unicos, conteos = np.unique(datos, return_counts=True)
frecuencia_observada = conteos / muestra

plt.figure(figsize=(12, 6))
plt.bar(valores_unicos, frecuencia_observada, alpha=0.7, label='Datos generados', color='skyblue', edgecolor='black')

# Calcular PMF teórica
x_teorico = np.arange(0, max(valores_unicos)+1)
pmf_teorica = poisson.pmf(x_teorico, mu=lambda_param)

plt.plot(x_teorico, pmf_teorica, 'ro-', markersize=5, label='PMF Teórica')
plt.title('Distribución de Poisson: Datos Generados vs Teórica')
plt.xlabel('Valor (k)')
plt.ylabel('Probabilidad')
plt.xticks(x_teorico)
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()

# Prueba de Chi-cuadrado de bondad de ajuste
# Calcular frecuencias observadas y esperadas
max_k = max(valores_unicos)
frec_obs, _ = np.histogram(datos, bins=np.arange(-0.5, max_k + 1.5))
rango_teorico = np.arange(0, len(frec_obs))  # Asegurar misma longitud que frec_obs
frec_esp = poisson.pmf(rango_teorico, mu=lambda_param) * muestra

# Agrupación corregid
umbral = 5
frec_obs_agrupada = []
frec_esp_agrupada = []
sum_obs_temp = 0
sum_esp_temp = 0

for k in range(len(frec_esp)):
    sum_obs_temp += frec_obs[k]
    sum_esp_temp += frec_esp[k]
    
    if sum_esp_temp >= umbral:
        frec_obs_agrupada.append(sum_obs_temp)
        frec_esp_agrupada.append(sum_esp_temp)
        sum_obs_temp = 0
        sum_esp_temp = 0


if sum_obs_temp > 0 or sum_esp_temp > 0:
    # Combinar con el último grupo si el residual es pequeño
    if len(frec_esp_agrupada) > 0:
        frec_obs_agrupada[-1] += sum_obs_temp
        frec_esp_agrupada[-1] += sum_esp_temp
    else:
        frec_obs_agrupada.append(sum_obs_temp)
        frec_esp_agrupada.append(sum_esp_temp)

# Ajuste final para coincidir con 1200
suma_obs = sum(frec_obs_agrupada)
suma_esp = sum(frec_esp_agrupada)

# Normalizar frec_esp_agrupada al tamaño de muestra
factor_ajuste = muestra / suma_esp
frec_esp_agrupada = [x * factor_ajuste for x in frec_esp_agrupada]

# Realizar prueba
chi2_stat, p_valor = chisquare(f_obs=frec_obs_agrupada, f_exp=frec_esp_agrupada)

print(f"Suma observada: {sum(frec_obs_agrupada)}")
print(f"Suma esperada: {sum(frec_esp_agrupada)}")
print(f"Estadístico Chi-cuadrado: {chi2_stat:.4f}")
print(f"Valor p: {p_valor:.4f}")


# Interpretación
alpha = 0.05
if p_valor > alpha:
    print("Conclusión: Los datos siguen una distribución de Poisson (no se rechaza H0)")
else:
    print("Conclusión: Los datos NO siguen una distribución de Poisson (se rechaza H0)")