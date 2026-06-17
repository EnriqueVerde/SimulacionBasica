import numpy as np
import matplotlib.pyplot as plt

# 1. Definimos la función objetivo f(x)
# Usaremos una curva en forma de campana (distribución normal no normalizada)
def f(x):
    return np.exp(-0.5 * (x - 5)**2)

def aceptacion_rechazo(N, a, b, M):
    """
    N: Número de muestras a aceptar
    a, b: Rango del eje X
    M: Cota superior (valor máximo en el eje Y)
    """
    aceptados_x = []
    aceptados_y = []
    rechazados_x = []
    rechazados_y = []

    # Iteramos hasta conseguir N muestras aceptadas
    while len(aceptados_x) < N:
        # Generamos R1 y R2
        R1 = np.random.uniform(0, 1)
        R2 = np.random.uniform(0, 1)
        
        # Mapeamos R1 al rango [a, b]
        X = a + (b - a) * R1
        
        # Evaluamos la condición del pizarrón: R2 < f(X) / M
        if R2 < (f(X) / M):
            aceptados_x.append(X)
            # Guardamos la coordenada Y (M * R2) solo para la visualización
            aceptados_y.append(M * R2) 
        else:
            rechazados_x.append(X)
            rechazados_y.append(M * R2)
            
    return aceptados_x, aceptados_y, rechazados_x, rechazados_y

# Parámetros basados en el esquema
a, b = 0, 10
M = 1.2  # Un valor por encima del pico de f(x) (que es 1.0)
N_muestras = 500

# Ejecutamos la simulación
ax, ay, rx, ry = aceptacion_rechazo(N_muestras, a, b, M)

# 3. Visualización
x_plot = np.linspace(a, b, 1000)
y_plot = f(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, 'k-', linewidth=2, label='f(x) - Función objetivo')
plt.axhline(M, color='gray', linestyle='--', label='M - Cota superior')

# Graficamos los puntos
plt.scatter(ax, ay, color='green', alpha=0.6, label='Aceptados', marker='.')
plt.scatter(rx, ry, color='red', alpha=0.3, label='Rechazados', marker='.')

plt.title('Método de Aceptación-Rechazo')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()