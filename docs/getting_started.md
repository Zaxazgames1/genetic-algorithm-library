# 🚀 Primeros Pasos con Genetic Algorithm Library

Esta guía te ayudará a comenzar a utilizar la librería Genetic Algorithm Library para resolver problemas de optimización mediante algoritmos genéticos y evolutivos.

## 📋 Contenido
- [Instalación](#instalación)
- [Conceptos Básicos](#conceptos-básicos)
- [Ejemplo Básico](#ejemplo-básico)
- [Codificaciones Genéticas](#codificaciones-genéticas)
- [Parámetros Principales](#parámetros-principales)
- [Visualización de Resultados](#visualización-de-resultados)
- [Optimización Multi-objetivo](#optimización-multi-objetivo)
- [Modelo de Islas](#modelo-de-islas)
- [Siguientes Pasos](#siguientes-pasos)

## 💻 Instalación

### Requisitos previos
- Python 3.6 o superior
- NumPy
- Matplotlib
- SciPy (opcional, para algunas funcionalidades avanzadas)

### Instalación desde PyPI
La forma más sencilla de instalar la librería es usando pip:

```bash
pip install genetic-algorithm-library
```

### Instalación desde el código fuente
También puedes instalar la librería directamente desde el repositorio:

```bash
git clone https://github.com/Zaxazgames1/genetic-algorithm-library.git
cd genetic-algorithm-library
pip install -e .
```

### Verificación de la instalación
Para verificar que la instalación es correcta, puedes ejecutar:

```python
# test_import.py
from genetic_algorithm import run_genetic_algorithm
print("Importación exitosa!")
```

## 🧠 Conceptos Básicos

### ¿Qué es un Algoritmo Genético?
Los algoritmos genéticos son métodos de optimización inspirados en la evolución natural y la genética. Funcionan evolucionando una población de soluciones candidatas (individuos) a través de generaciones, aplicando selección, cruce y mutación para encontrar soluciones cada vez mejores.

### Componentes principales
1. **Población**: Conjunto de soluciones candidatas (individuos)
2. **Función de aptitud (fitness)**: Evalúa la calidad de cada solución
3. **Selección**: Escoge los mejores individuos para reproducción
4. **Cruce**: Combina dos soluciones para crear descendencia
5. **Mutación**: Introduce pequeños cambios aleatorios
6. **Reemplazo**: Determina qué individuos pasan a la siguiente generación

### Ciclo del algoritmo genético
1. Inicializar población aleatoria
2. Evaluar la aptitud de cada individuo
3. Seleccionar padres para reproducción
4. Aplicar cruce para generar descendencia
5. Aplicar mutación a la descendencia
6. Evaluar la aptitud de la nueva generación
7. Reemplazar la población anterior
8. Repetir pasos 3-7 hasta alcanzar un criterio de parada

## 🔍 Ejemplo Básico

Este ejemplo ilustra cómo resolver un problema simple de optimización:

```python
from genetic_algorithm import run_genetic_algorithm, plot_evolution
import numpy as np

# Definir función objetivo (maximizar el negativo para minimizar la función)
def objective_function(x):
    # Función Sphere: suma de cuadrados (busca mínimo en [0,0,...,0])
    return -np.sum(x**2)

# Configurar y ejecutar el algoritmo genético
result = run_genetic_algorithm(
    objective_function=objective_function,  # Función objetivo
    gene_length=3,                          # Dimensión del problema
    bounds=(-10, 10),                       # Límites de búsqueda
    pop_size=50,                            # Tamaño de población
    num_generations=100,                    # Número de generaciones
    selection_type="tournament",            # Método de selección
    crossover_type="blend",                 # Tipo de cruce
    mutation_type="gaussian",               # Tipo de mutación
    adaptive=True,                          # Adaptación dinámica
    verbose=True                            # Mostrar progreso
)

# Mostrar resultados
print(f"Mejor solución encontrada: {result['best_individual']}")
print(f"Mejor fitness: {result['best_fitness']}")

# Visualizar la evolución
plot_evolution(result['history'])
```

## 🧬 Codificaciones Genéticas

La librería soporta diferentes codificaciones según el tipo de problema:

### Codificación Real (valores continuos)
```python
population = create_population(size=50, gene_length=5, min_val=-5, max_val=5, encoding="real")
```
Ideal para optimización de funciones continuas y problemas con variables reales.

### Codificación Binaria (0-1)
```python
population = create_population(size=50, gene_length=8, encoding="binary")
```
Útil para problemas con decisiones binarias, como selección de características o problema de la mochila.

### Codificación Entera
```python
population = create_population(size=50, gene_length=5, min_val=1, max_val=10, encoding="integer")
```
Para problemas con variables discretas como asignación de recursos o planificación.

### Codificación de Permutación
```python
population = create_population(size=50, gene_length=10, encoding="permutation")
```
Ideal para problemas como el TSP (Problema del Viajante), secuenciación de tareas o scheduling.

## ⚙️ Parámetros Principales

### Tamaño de población
El parámetro `pop_size` controla el número de soluciones candidatas en cada generación:
- **Valores pequeños** (20-50): Ejecución más rápida, pero menos diversidad
- **Valores grandes** (100-500): Mayor exploración pero más costoso computacionalmente
- **Recomendación**: 10 veces la dimensión del problema como punto de partida

### Número de generaciones
El parámetro `num_generations` determina cuántas iteraciones ejecutará el algoritmo:
- Comienza con valores moderados (50-200) y aumenta si es necesario
- Utiliza `early_stopping` para detener automáticamente cuando la mejora se estanca

### Métodos de selección
Diferentes estrategias para elegir padres:
- **tournament**: Selección por torneo (robusta y versátil)
- **roulette**: Selección proporcional al fitness (presión selectiva variable)
- **rank**: Selección basada en rangos (reduce dominancia de superindividuos)
- **sus**: Muestreo Universal Estocástico (más equitativo)
- **boltzmann**: Selección con "temperatura" (ajusta presión selectiva dinámicamente)

### Operadores de cruce
Métodos para combinar soluciones:
- **uniform**: Uniforme (intercambio de genes con igual probabilidad)
- **single_point**: Un punto (divide genoma en dos partes)
- **two_point**: Dos puntos (intercambia segmento central)
- **blend**: Mezcla (combina valores numéricamente, solo para codificación real)
- **sbx**: Cruce Binario Simulado (preserva distancias, ideal para problemas reales)
- **pmx**: Partially Mapped Crossover (para problemas de permutación)

### Operadores de mutación
Estrategias para introducir variación:
- **gaussian**: Mutación gaussiana (añade ruido normal, ideal para valores reales)
- **uniform**: Mutación uniforme (reemplaza con valores aleatorios uniformes)
- **reset**: Mutación de reseteo (asigna valores extremos)
- **swap**: Intercambio (permuta posiciones, para permutaciones)
- **inversion**: Inversión (invierte segmento, para permutaciones)

### Adaptación dinámica
Cuando `adaptive=True`, la librería ajusta automáticamente:
- Tasas de mutación según rendimiento
- Parámetros de los operadores según diversidad
- Presión selectiva según convergencia

## 📊 Visualización de Resultados

La librería ofrece varias herramientas de visualización:

### Evolución del fitness
```python
from genetic_algorithm import plot_evolution
plot_evolution(result['history'])
```

### Diversidad poblacional
```python
from genetic_algorithm.utils.visualization import plot_population_diversity
plot_population_diversity(result['history'])
```

### Animación de la evolución
```python
from genetic_algorithm.utils.visualization import animate_evolution
animate_evolution(result['history'], interval=200, save_path="evolucion.gif")
```

## 🎯 Optimización Multi-objetivo

Para problemas con múltiples objetivos conflictivos:

```python
from genetic_algorithm import run_multi_objective_ga, plot_pareto_front

# Definir dos funciones objetivo contrapuestas
def objective1(x):
    return -np.sum(x**2)  # Minimizar suma de cuadrados

def objective2(x):
    return -np.sum((x-2)**2)  # Minimizar distancia a [2,2]

# Ejecutar algoritmo multi-objetivo
result = run_multi_objective_ga(
    objective_functions=[objective1, objective2],
    gene_length=2,
    bounds=(-5, 5),
    pop_size=100,
    num_generations=50
)

# Visualizar frente de Pareto
plot_pareto_front(
    result['pareto_fitness'],
    objective_names=["Objetivo 1", "Objetivo 2"]
)
```

### Entendiendo el frente de Pareto
- Las soluciones en el frente de Pareto representan compromisos óptimos
- No existe una solución que mejore todos los objetivos simultáneamente
- La visualización ayuda a elegir el balance adecuado entre objetivos

## 🏝️ Modelo de Islas

Para problemas complejos, el modelo de islas implementa evolución en poblaciones separadas:

```python
from genetic_algorithm import run_island_model_ga

result = run_island_model_ga(
    objective_function=objective_function,
    gene_length=5,
    bounds=(-10, 10),
    num_islands=4,              # Número de islas
    pop_size_per_island=30,     # Población por isla
    num_generations=50,
    migration_interval=10,      # Frecuencia de migración
    migration_rate=0.1,         # Proporción que migra
    selection_types=["tournament", "roulette", "rank", "sus"],  # Un método por isla
    crossover_types=["uniform", "blend", "two_point", "sbx"],   # Un tipo por isla
)
```

### Ventajas del modelo de islas
- Mantiene mayor diversidad genética
- Evita convergencia prematura
- Permite explorar diferentes regiones del espacio simultáneamente
- Aprovecha paralelismo en sistemas multi-núcleo

## 🔜 Siguientes Pasos

Una vez que domines los conceptos básicos, puedes:

1. **Explorar problemas más complejos** en los ejemplos: `examples/tsp_example.py`, `examples/multi_objective.py`
2. **Implementar tus propios operadores** extendiendo las clases base
3. **Optimizar el rendimiento** mediante paralelización o compilación JIT
4. **Integrar con otros frameworks** como scikit-learn, TensorFlow o PyTorch
5. **Consultar la documentación completa** en la [API Reference](api_reference.md)

---

Si tienes dudas, problemas o sugerencias, no dudes en crear un issue en nuestro [repositorio de GitHub](https://github.com/Zaxazgames1/genetic-algorithm-library/issues).