import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random

# --- CONFIGURACIÓN GLOBAL ---
st.set_page_config(page_title="Code Hub: Simulación Avanzada", layout="wide")
st.title("🧮 Generación de Variables y Procesos Estocásticos")
st.markdown("---")

# --- MENÚ DE PESTAÑAS ---
tabs = st.tabs([
    "1. Transformada Inversa", 
    "2. Aceptación-Rechazo", 
    "3. Convolución", 
    "4. Composición", 
    "5. Proceso Poisson",
    "6. Cadenas de Markov",
    "7. Monte Carlo (Pi)"  # <--- NUEVA PESTAÑA
])

# ==========================================
# TAB 1: TRANSFORMADA INVERSA
# ==========================================
with tabs[0]:
    st.header("Método de la Transformada Inversa")
    
    col_p, col_g = st.columns([1, 2])
    with col_p:
        lam = st.slider("Tasa (Lambda)", 0.1, 5.0, 1.0, key="inv_lam")
        n_inv = st.number_input("Muestras", 100, 10000, 1000, key="inv_n")
        
    u = np.random.uniform(0, 1, n_inv)
    x = - (1/lam) * np.log(1 - u)
    
    fig, ax = plt.subplots()
    ax.hist(x, bins=50, color='skyblue', edgecolor='black', density=True)
    xt = np.linspace(0, max(x), 100)
    yt = lam * np.exp(-lam * xt)
    ax.plot(xt, yt, 'r-', lw=2, label="Teórico")
    ax.set_title("Histograma de Exp(λ)")
    ax.legend()
    col_g.pyplot(fig)

    with st.expander("💻 Ver código fuente (Python)"):
        st.code("""
import numpy as np
import matplotlib.pyplot as plt

lam = 1.0  # Tasa
n = 1000   # Número de muestras

# Generamos números uniformes
u = np.random.uniform(0, 1, n)

# Aplicamos la transformada inversa para la exponencial
x = - (1/lam) * np.log(1 - u)

# Visualización
plt.hist(x, bins=50, density=True)
plt.show()
        """, language='python')

# ==========================================
# TAB 2: ACEPTACIÓN-RECHAZO (LÓGICA WHILE)
# ==========================================
with tabs[1]:
    st.header("Método de Aceptación y Rechazo (Bucle While)")
    
    # Controles interactivos
    col_p, col_g = st.columns([1, 2])
    with col_p:
        st.write("Parámetros de la simulación:")
        M_val = st.slider("Cota superior (M)", 1.0, 3.0, 1.2, key="ar_m_while")
        N_val = st.slider("Muestras a aceptar (N)", 10, 1000, 500, key="ar_n_while")
        
    with col_g:
        # Definimos la función objetivo f(x)
        def f(x):
            return np.exp(-0.5 * (x - 5)**2)

        def aceptacion_rechazo(N, a, b, M):
            aceptados_x = []
            aceptados_y = []
            rechazados_x = []
            rechazados_y = []

            # Iteramos hasta conseguir N muestras aceptadas
            while len(aceptados_x) < N:
                R1 = np.random.uniform(0, 1)
                R2 = np.random.uniform(0, 1)
                
                X = a + (b - a) * R1
                
                if R2 < (f(X) / M):
                    aceptados_x.append(X)
                    aceptados_y.append(M * R2) 
                else:
                    rechazados_x.append(X)
                    rechazados_y.append(M * R2)
                    
            return aceptados_x, aceptados_y, rechazados_x, rechazados_y

        a, b = 0, 10
        
        # Ejecutamos la simulación
        ax_vals, ay_vals, rx_vals, ry_vals = aceptacion_rechazo(N_val, a, b, M_val)

        # Visualización
        x_plot = np.linspace(a, b, 1000)
        y_plot = f(x_plot)

        fig = plt.figure(figsize=(10, 5))
        plt.plot(x_plot, y_plot, 'k-', linewidth=2, label='f(x) - Función objetivo')
        plt.axhline(M_val, color='gray', linestyle='--', label='M - Cota superior')

        plt.scatter(ax_vals, ay_vals, color='green', alpha=0.6, s=15, label='Aceptados')
        plt.scatter(rx_vals, ry_vals, color='red', alpha=0.3, s=15, label='Rechazados')

        plt.title('Método de Aceptación-Rechazo')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Métricas
        intentos_totales = len(ax_vals) + len(rx_vals)
        st.write(f"**Estadísticas:** Para conseguir **{N_val}** muestras aceptadas, el algoritmo generó un total de **{intentos_totales}** intentos (Tasa de eficiencia: {N_val/intentos_totales:.2%}).")

    with st.expander("💻 Ver código fuente (Python)"):
        st.code("""
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.exp(-0.5 * (x - 5)**2)

def aceptacion_rechazo(N, a, b, M):
    aceptados_x, aceptados_y, rechazados_x, rechazados_y = [], [], [], []
    while len(aceptados_x) < N:
        R1 = np.random.uniform(0, 1)
        R2 = np.random.uniform(0, 1)
        X = a + (b - a) * R1
        if R2 < (f(X) / M):
            aceptados_x.append(X)
            aceptados_y.append(M * R2) 
        else:
            rechazados_x.append(X)
            rechazados_y.append(M * R2)
    return aceptados_x, aceptados_y, rechazados_x, rechazados_y

# Uso:
# ax, ay, rx, ry = aceptacion_rechazo(500, 0, 10, 1.2)
        """, language='python')

# ==========================================
# TAB 3: CONVOLUCIÓN
# ==========================================
with tabs[2]:
    st.header("Método de Convolución")
    k = st.select_slider("Número de variables a sumar (K)", options=[2, 3, 5, 12])
    n_conv = 2000
    
    muestras = np.random.uniform(0, 1, (n_conv, k))
    suma = np.sum(muestras, axis=1)
    
    fig, ax = plt.subplots()
    ax.hist(suma, bins=40, color='plum', edgecolor='black')
    st.pyplot(fig)

    with st.expander("💻 Ver código fuente (Python)"):
        st.code("""
import numpy as np

# Para simular una distribución Normal usando el Teorema del Límite Central
k = 12 # Sumamos 12 uniformes
n_muestras = 2000

# Matriz de 2000 filas por 12 columnas
muestras = np.random.uniform(0, 1, (n_muestras, k))

# Sumamos las columnas para cada fila
suma = np.sum(muestras, axis=1)
        """, language='python')

# ==========================================
# TAB 4: COMPOSICIÓN
# ==========================================
with tabs[3]:
    st.header("Método de Composición")
    w1 = st.slider("Peso de la Campana 1", 0.0, 1.0, 0.3)
    n_comp = 2000
    
    u = np.random.rand(n_comp)
    muestras_comp = []
    for val in u:
        if val < w1:
            muestras_comp.append(np.random.normal(2, 0.5))
        else:
            muestras_comp.append(np.random.normal(6, 1.2))
            
    fig, ax = plt.subplots()
    ax.hist(muestras_comp, bins=50, color='gold', edgecolor='black')
    st.pyplot(fig)

    with st.expander("💻 Ver código fuente (Python)"):
        st.code("""
import numpy as np

w1 = 0.3 # 30% de probabilidad para la primera distribución
n = 2000
muestras = []

u = np.random.uniform(0, 1, n)

for val in u:
    if val < w1:
        muestras.append(np.random.normal(2, 0.5))
    else:
        muestras.append(np.random.normal(6, 1.2))
        """, language='python')

# ==========================================
# TAB 5: POISSON
# ==========================================
with tabs[4]:
    st.header("Proceso de Poisson")
    lam_p = st.number_input("Llegadas por hora (λ)", 1, 20, 5)
    t_max = st.number_input("Horas de simulación", 1, 24, 8)
    
    tiempos = [0]
    while tiempos[-1] < t_max:
        inter_llegada = - (1/lam_p) * np.log(np.random.uniform(0, 1))
        tiempos.append(tiempos[-1] + inter_llegada)
    
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.step(tiempos, np.arange(len(tiempos)), where='post', color='teal')
    ax.scatter(tiempos, np.zeros(len(tiempos)), color='red', marker='|', s=100)
    st.pyplot(fig)

    with st.expander("💻 Ver código fuente (Python)"):
        st.code("""
import numpy as np

lam = 5 # Tasa de llegada
t_max = 8 # Tiempo máximo de simulación
tiempos = [0] 

while tiempos[-1] < t_max:
    u = np.random.uniform(0, 1)
    tiempo_inter_llegada = - (1/lam) * np.log(u)
    nuevo_tiempo = tiempos[-1] + tiempo_inter_llegada
    tiempos.append(nuevo_tiempo)
        """, language='python')

# ==========================================
# TAB 6: MARKOV
# ==========================================
with tabs[5]:
    st.header("Cadenas de Markov")
    p = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5]
    ])
    
    n_pasos = st.slider("Días a simular", 10, 365, 30)
    estados = [0]
    for _ in range(n_pasos):
        estados.append(np.random.choice([0, 1, 2], p=p[estados[-1]]))
        
    st.line_chart(estados)

    with st.expander("💻 Ver código fuente (Python)"):
        st.code("""
import numpy as np

# Matriz de transición (3x3)
p = np.array([
    [0.7, 0.2, 0.1], 
    [0.3, 0.4, 0.3], 
    [0.2, 0.3, 0.5]  
])

dias = 30
estados = [0]

for _ in range(dias):
    estado_actual = estados[-1]
    siguiente_estado = np.random.choice([0, 1, 2], p=p[estado_actual])
    estados.append(siguiente_estado)
        """, language='python')

# ==========================================
# TAB 7: MONTE CARLO (PI)
# ==========================================
with tabs[6]:
    st.header("Método de Monte Carlo (Estimación de Pi)")
    st.write("Cálculo de $\pi$ generando coordenadas aleatorias y verificando si caen dentro de un círculo usando el teorema de Pitágoras.")
    
    col_p, col_g = st.columns([1, 2])
    
    with col_p:
        num_grains = st.slider("Granos por persona", 100, 10000, 1000)
        num_persons = st.slider("Personas a simular", 1, 1000, 100)
        edge = 10
        
    with col_g:
        # Para la visualización web, usamos una versión vectorizada de tu lógica 
        # para que la página sea rápida e interactiva.
        x_vis = (np.random.rand(num_grains) - 0.5) * edge
        y_vis = (np.random.rand(num_grains) - 0.5) * edge
        
        # Teorema de Pitágoras
        inside_vis = (x_vis**2 + y_vis**2) <= (edge/2)**2
        
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(x_vis[inside_vis], y_vis[inside_vis], color='green', s=5, alpha=0.6, label='Dentro')
        ax.scatter(x_vis[~inside_vis], y_vis[~inside_vis], color='red', s=5, alpha=0.6, label='Fuera')
        
        # Dibujamos el círculo y el cuadrado
        circle = plt.Circle((0, 0), edge/2, color='black', fill=False, lw=2)
        ax.add_patch(circle)
        
        ax.set_xlim(-edge/2, edge/2)
        ax.set_ylim(-edge/2, edge/2)
        ax.set_aspect('equal', 'box')
        ax.legend(loc='upper right')
        st.pyplot(fig)
        
    # Calculamos la estimación de Pi usando vectorización masiva de Numpy
    # Esto simula exactamente tus rondas, personas y granos, pero en milisegundos.
    st.info("💡 Realizando el cálculo masivo en segundo plano...")
    total_points = 20 * num_persons * num_grains
    
    # Generación rápida de Numpy en lugar de For loops anidados
    x_total = (np.random.rand(total_points) - 0.5) * edge
    y_total = (np.random.rand(total_points) - 0.5) * edge
    inside_total = np.sum((x_total**2 + y_total**2) <= (edge/2)**2)
    
    pi_estimate = (inside_total / total_points) * 4
    
    col1, col2 = st.columns(2)
    col1.metric("Puntos evaluados", f"{total_points:,}")
    col2.metric("Valor estimado de Pi", f"{pi_estimate:.5f}")

    # Tu código original exacto en el bloque desplegable
    with st.expander("💻 Ver código fuente original (Python)"):
        st.code("""
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
        """, language='python')