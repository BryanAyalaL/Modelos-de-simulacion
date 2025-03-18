from scipy.stats import binom

# Parámetros
muestra = 6
p = 0.75

# a) Probabilidad de exactamente 4 éxitos
prob_a = (binom.pmf(4, muestra, p))*100

# b) Probabilidad de al menos 3 éxitos (3, 4, 5, 6)
prob_b =( 1 - binom.cdf(2, muestra, p))*100
print(f"a) {prob_a:.0f}")  # Resultado: 29.66
print(f"b) {prob_b:.0f}")  # Resultado: 96.24