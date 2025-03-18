from scipy.stats import norm

# Parámetros
mu = 500    # Media
sigma = 30  # Desviación estándar

# a) Probabilidad de que sea menor que 500 horas (es la media)
prob_a = norm.cdf(500, mu, sigma)
# b) Probabilidad entre 480 y 520 horas
prob_b = norm.cdf(520, mu, sigma) - norm.cdf(480, mu, sigma)
# c) Probabilidad entre 500 y 510 horas
prob_c = norm.cdf(510, mu, sigma) - norm.cdf(500, mu, sigma)
# d) Probabilidad entre 470 y 500 horas
prob_d = norm.cdf(500, mu, sigma) - norm.cdf(470, mu, sigma)
# e) Probabilidad entre 450 y 490 horas
prob_e = norm.cdf(490, mu, sigma) - norm.cdf(450, mu, sigma)

print(f"a) {100*prob_a:.0f}")  # a) 0.5000
print(f"b) {100*prob_b:.0f}")  # b) 0.4950
print(f"c) {100*prob_c:.0f}")  # c) 0.1293
print(f"d) {100*prob_d:.0f}")  # d) 0.3413
print(f"e) {100*prob_e:.0f}")  # e) 0.2286