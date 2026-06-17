import random
import numpy as np

# Fijamos la semilla para que los resultados sean reproducibles
random.seed(2021)

pi_values = list()
num_persons = 1000
num_rounds = 20
num_grains = 1000
edge = 10  # Tamaño del lado del cuadrado

# Ejecutamos la simulación
for r in range(num_rounds):
    for p in range(num_persons):
        in_circle = 0
        for g in range(num_grains):
            # Generamos coordenadas aleatorias x, y centradas en el origen (0,0)
            x = (random.random() - 0.5) * edge
            y = (random.random() - 0.5) * edge
            
            # Comprobamos si el punto cae dentro del círculo usando el teorema de Pitágoras
            if x**2 + y**2 <= (edge/2)**2:
                in_circle += 1
                
        # Calculamos la estimación de pi para esta persona y ronda
        pi = (in_circle / num_grains) * 4
        pi_values.append(pi)

# Imprimimos el promedio de todas las simulaciones (Debería ser muy cercano a 3.14159...)
print(f"Valor estimado de Pi: {np.mean(pi_values)}")