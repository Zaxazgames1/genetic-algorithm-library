# 📚 Referencia de API - Genetic Algorithm Library

<div align="center">
  <img src="https://raw.githubusercontent.com/Zaxazgames1/genetic-algorithm-library/main/docs/images/logo.png" alt="Genetic Algorithm Library Logo" width="200"/>
  <br>
  <em>Documentación técnica completa de la librería</em>
</div>

## Índice de Contenidos

- [Funciones Principales](#funciones-principales)
- [Módulos Core](#módulos-core)
  - [Population](#population)
  - [Selection](#selection)
  - [Crossover](#crossover)
  - [Mutation](#mutation)
  - [Fitness](#fitness)
- [Módulo Problems](#módulo-problems)
  - [Continuous](#continuous)
  - [Discrete](#discrete)
  - [Combinatorial](#combinatorial)
- [Módulo Utils](#módulo-utils)
  - [Visualization](#visualization)
  - [Helpers](#helpers)
- [Ejemplos de Uso](#ejemplos-de-uso)

---

## Funciones Principales

Estas funciones proporcionan la interfaz de alto nivel de la librería, permitiendo ejecutar algoritmos genéticos completos con una sola llamada.

### `run_genetic_algorithm()`

Ejecuta un algoritmo genético estándar para optimización de un solo objetivo.

```python
def run_genetic_algorithm(
    objective_function, 
    gene_length, 
    bounds=(0, 1), 
    pop_size=100, 
    num_generations=100, 
    selection_type="tournament", 
    crossover_type="uniform",
    mutation_type="gaussian", 
    mutation_rate=0.01,
    encoding="real", 
    adaptive=True, 
    elitism=True,
    verbose=True, 
    early_stopping=None, 
    callbacks=None
)
```

**Parámetros:**
- `objective_function` (callable): Función objetivo a maximizar
- `gene_length` (int): Longitud del genoma de cada individuo
- `bounds` (tuple): Límites de valores para los genes (min, max)
- `pop_size` (int): Tamaño de la población
- `num_generations` (int): Número de generaciones a ejecutar
- `selection_type` (str): Método de selección ["tournament", "roulette", "rank", "sus", "boltzmann"]
- `crossover_type` (str): Tipo de cruce ["uniform", "single_point", "two_point", "blend", "sbx", "pmx"]
- `mutation_type` (str): Tipo de mutación ["gaussian", "uniform", "reset", "swap", "inversion"]
- `mutation_rate` (float): Tasa inicial de mutación
- `encoding` (str): Tipo de codificación ["real", "binary", "integer", "permutation"]
- `adaptive` (bool): Si es True, la tasa de mutación se adapta durante la ejecución
- `elitism` (bool): Si es True, conserva el mejor individuo en cada generación
- `verbose` (bool): Si es True, muestra información durante la ejecución
- `early_stopping` (int): Detener el algoritmo si no hay mejora después de X generaciones
- `callbacks` (list): Lista de funciones a llamar en cada generación

**Retorna:**
- `dict`: Diccionario con:
  - `best_individual`: Mejor solución encontrada
  - `best_fitness`: Valor de fitness de la mejor solución
  - `history`: Historial de la evolución
  - `generations`: Número de generaciones ejecutadas
  - `execution_time`: Tiempo total de ejecución
  - `final_diversity`: Diversidad final de la población

**Ejemplo:**
```python
def objective_function(x):
    return -(x[0]**2 + x[1]**2)  # Minimizar suma de cuadrados

result = run_genetic_algorithm(
    objective_function=objective_function,
    gene_length=2,
    bounds=(-10, 10),
    pop_size=50,
    num_generations=100
)
```

### `run_multi_objective_ga()`

Ejecuta un algoritmo genético para optimización multi-objetivo.

```python
def run_multi_objective_ga(
    objective_functions, 
    gene_length, 
    bounds=(0, 1),
    pop_size=100, 
    num_generations=100,
    selection_type="tournament", 
    crossover_type="uniform",
    mutation_type="gaussian", 
    mutation_rate=0.01,
    encoding="real", 
    verbose=True
)
```

**Parámetros:**
- `objective_functions` (list): Lista de funciones objetivo
- `gene_length` (int): Longitud del genoma de cada individuo
- `bounds` (tuple): Límites de valores para los genes (min, max)
- `pop_size` (int): Tamaño de la población
- `num_generations` (int): Número de generaciones a ejecutar
- `selection_type` (str): Método de selección
- `crossover_type` (str): Tipo de cruce
- `mutation_type` (str): Tipo de mutación
- `mutation_rate` (float): Tasa de mutación
- `encoding` (str): Tipo de codificación
- `verbose` (bool): Si es True, muestra información durante la ejecución

**Retorna:**
- `dict`: Diccionario con:
  - `pareto_front`: Conjunto de soluciones no dominadas
  - `pareto_fitness`: Valores de fitness para las soluciones no dominadas
  - `history`: Historial de evolución

**Ejemplo:**
```python
def objective1(x):
    return -sum(x**2)  # Minimizar suma de cuadrados

def objective2(x):
    return -sum((x-2)**2)  # Minimizar distancia a (2,2)

result = run_multi_objective_ga(
    objective_functions=[objective1, objective2],
    gene_length=2,
    bounds=(-5, 5),
    pop_size=100
)
```

### `run_island_model_ga()`

Ejecuta un algoritmo genético con modelo de islas para preservar la diversidad.

```python
def run_island_model_ga(
    objective_function, 
    gene_length, 
    bounds=(0, 1),
    num_islands=4, 
    pop_size_per_island=50, 
    num_generations=100,
    migration_interval=10, 
    migration_rate=0.1,
    selection_types=None, 
    crossover_types=None, 
    mutation_types=None,
    encoding="real", 
    verbose=True
)
```

**Parámetros:**
- `objective_function` (callable): Función objetivo a maximizar
- `gene_length` (int): Longitud del genoma de cada individuo
- `bounds` (tuple): Límites de valores para los genes (min, max)
- `num_islands` (int): Número de islas (subpoblaciones)
- `pop_size_per_island` (int): Tamaño de población en cada isla
- `num_generations` (int): Número de generaciones a ejecutar
- `migration_interval` (int): Intervalo de generaciones entre migraciones
- `migration_rate` (float): Proporción de individuos que migran
- `selection_types` (list): Lista de métodos de selección para cada isla
- `crossover_types` (list): Lista de tipos de cruce para cada isla
- `mutation_types` (list): Lista de tipos de mutación para cada isla
- `encoding` (str): Tipo de codificación
- `verbose` (bool): Si es True, muestra información durante la ejecución

**Retorna:**
- `dict`: Diccionario con:
  - `best_individual`: Mejor solución global encontrada
  - `best_fitness`: Valor de fitness de la mejor solución
  - `history`: Historial de evolución
  - `final_islands`: Estado final de cada isla

**Ejemplo:**
```python
result = run_island_model_ga(
    objective_function=my_objective,
    gene_length=5,
    bounds=(-10, 10),
    num_islands=4,
    pop_size_per_island=50,
    migration_interval=10
)
```

---

## Módulos Core

Estos módulos contienen los componentes fundamentales para construir algoritmos genéticos personalizados.

### Population

Funciones para crear y gestionar poblaciones de individuos.

#### `create_population()`

```python
def create_population(size, gene_length, min_val=0, max_val=1, encoding="real")
```

Crea una población inicial de individuos con diferentes tipos de codificación.

**Parámetros:**
- `size` (int): Tamaño de la población
- `gene_length` (int): Longitud del genoma de cada individuo
- `min_val` (float): Valor mínimo para los genes
- `max_val` (float): Valor máximo para los genes
- `encoding` (str): Tipo de codificación ["real", "binary", "integer", "permutation"]

**Retorna:**
- `ndarray`: Población inicial de tamaño (size, gene_length)

#### `initialize_from_samples()`

```python
def initialize_from_samples(samples, size, noise=0.1)
```

Inicializa una población a partir de muestras conocidas, útil para partir de soluciones aproximadas.

**Parámetros:**
- `samples` (ndarray): Muestras conocidas que guiarán la inicialización
- `size` (int): Tamaño de la población a crear
- `noise` (float): Cantidad de ruido a añadir a las muestras (variabilidad)

**Retorna:**
- `ndarray`: Población inicial basada en las muestras

#### `check_population_diversity()`

```python
def check_population_diversity(population, threshold=0.01)
```

Evalúa la diversidad de la población actual.

**Parámetros:**
- `population` (ndarray): Población a evaluar
- `threshold` (float): Umbral para considerar dos individuos como similares

**Retorna:**
- `float`: Índice de diversidad (0-1, donde 1 es máxima diversidad)

### Selection

Métodos de selección para identificar individuos prometedores para reproducción.

#### `selection()`

```python
def selection(population, fitness_values, num_parents, selection_type="tournament", tournament_size=3)
```

Selecciona individuos para reproducción utilizando diferentes métodos.

**Parámetros:**
- `population` (ndarray): Población actual
- `fitness_values` (ndarray): Valores de aptitud correspondientes a la población
- `num_parents` (int): Número de padres a seleccionar
- `selection_type` (str): Método de selección ["tournament", "roulette", "rank", "sus", "boltzmann"]
- `tournament_size` (int): Tamaño del torneo para selección por torneo

**Retorna:**
- `ndarray`: Padres seleccionados

### Crossover

Operadores de cruce para generar nuevos individuos combinando padres seleccionados.

#### `crossover()`

```python
def crossover(parents, offspring_size, crossover_type="uniform", crossover_rate=0.8)
```

Realiza el cruce entre padres para crear descendencia.

**Parámetros:**
- `parents` (ndarray): Arreglo de padres seleccionados
- `offspring_size` (tuple): Tamaño de la descendencia: (n_offspring, n_genes)
- `crossover_type` (str): Tipo de cruce ["uniform", "single_point", "two_point", "blend", "sbx", "pmx"]
- `crossover_rate` (float): Probabilidad de que ocurra el cruce

**Retorna:**
- `ndarray`: Descendencia generada

#### `crossover_permutation()`

```python
def crossover_permutation(parents, offspring_size, crossover_type="pmx")
```

Función especializada para cruce de permutaciones (problemas como TSP).

**Parámetros:**
- `parents` (ndarray): Arreglo de padres seleccionados (permutaciones)
- `offspring_size` (tuple): Tamaño de la descendencia: (n_offspring, n_genes)
- `crossover_type` (str): Tipo de cruce ["pmx", "ox", "cx"]

**Retorna:**
- `ndarray`: Descendencia generada (permutaciones válidas)

### Mutation

Operadores de mutación para mantener diversidad en la población.

#### `mutation()`

```python
def mutation(offspring, mutation_rate=0.01, mutation_type="gaussian", min_val=0, max_val=1, encoding="real")
```

Aplica mutación a la descendencia.

**Parámetros:**
- `offspring` (ndarray): Descendencia a mutar
- `mutation_rate` (float): Tasa de mutación (probabilidad de que un gen mute)
- `mutation_type` (str): Tipo de mutación ["gaussian", "uniform", "reset", "swap", "inversion"]
- `min_val` (float): Valor mínimo para los genes
- `max_val` (float): Valor máximo para los genes
- `encoding` (str): Tipo de codificación

**Retorna:**
- `ndarray`: Descendencia mutada

#### `adaptive_mutation()`

```python
def adaptive_mutation(offspring, fitness_values, best_fitness, avg_fitness, min_val=0, max_val=1, base_rate=0.01, encoding="real")
```

Aplica mutación adaptativa basada en el fitness.

**Parámetros:**
- `offspring` (ndarray): Descendencia a mutar
- `fitness_values` (ndarray): Valores de aptitud correspondientes
- `best_fitness` (float): Mejor valor de fitness en la población
- `avg_fitness` (float): Valor promedio de fitness en la población
- `min_val` (float): Valor mínimo para los genes
- `max_val` (float): Valor máximo para los genes
- `base_rate` (float): Tasa base de mutación
- `encoding` (str): Tipo de codificación

**Retorna:**
- `ndarray`: Descendencia mutada con tasa adaptativa

#### `self_adaptation()`

```python
def self_adaptation(offspring, mutation_params, learning_rate=0.1)
```

Implementa auto-adaptación donde los parámetros de mutación evolucionan junto con los individuos.

**Parámetros:**
- `offspring` (ndarray): Descendencia a mutar
- `mutation_params` (ndarray): Parámetros de mutación actuales para cada individuo
- `learning_rate` (float): Tasa de aprendizaje para la adaptación

**Retorna:**
- `tuple`: (offspring_mutada, nuevos_parametros_mutacion)

### Fitness

Funciones para evaluar la aptitud de los individuos y manejar restricciones.

#### `fitness_function()`

```python
def fitness_function(individual, objective_function)
```

Evalúa la aptitud de un individuo.

**Parámetros:**
- `individual` (array-like): Individuo a evaluar
- `objective_function` (callable): Función objetivo que determina la aptitud

**Retorna:**
- `float`: Valor de aptitud del individuo

#### `rank_fitness()`

```python
def rank_fitness(fitness_values)
```

Convierte valores de fitness a rangos. Útil para problemas donde la escala de fitness puede variar mucho.

**Parámetros:**
- `fitness_values` (ndarray): Valores originales de aptitud

**Retorna:**
- `ndarray`: Valores de aptitud convertidos a rangos

#### `constrained_fitness()`

```python
def constrained_fitness(individual, objective_function, constraint_functions, penalty_factor=1e6)
```

Evalúa fitness con manejo de restricciones mediante penalización.

**Parámetros:**
- `individual` (array-like): Individuo a evaluar
- `objective_function` (callable): Función objetivo principal
- `constraint_functions` (list): Lista de funciones de restricción que deben ser >= 0 para ser válidas
- `penalty_factor` (float): Factor de penalización para restricciones violadas

**Retorna:**
- `float`: Valor de aptitud penalizado

#### `multi_objective_fitness()`

```python
def multi_objective_fitness(individual, objective_functions)
```

Evalúa fitness en problemas multi-objetivo.

**Parámetros:**
- `individual` (array-like): Individuo a evaluar
- `objective_functions` (list): Lista de funciones objetivo

**Retorna:**
- `ndarray`: Vector de valores de aptitud para cada objetivo

#### `pareto_dominance()`

```python
def pareto_dominance(fitness1, fitness2)
```

Determina si una solución domina a otra en el sentido de Pareto.

**Parámetros:**
- `fitness1` (array-like): Vector de fitness de la primera solución
- `fitness2` (array-like): Vector de fitness de la segunda solución

**Retorna:**
- `bool`: True si fitness1 domina a fitness2, False en caso contrario

#### `get_pareto_front()`

```python
def get_pareto_front(population, fitness_values)
```

Obtiene el frente de Pareto de una población.

**Parámetros:**
- `population` (ndarray): Población de individuos
- `fitness_values` (ndarray): Matriz de valores de fitness (individuos x objetivos)

**Retorna:**
- `tuple`: (índices del frente de Pareto, individuos en el frente, valores de fitness)

---

## Módulo Problems

Este módulo contiene funciones para problemas específicos y funciones de prueba estándar.

### Continuous

Funciones de prueba estándar para optimización continua.

#### Funciones implementadas:

- `sphere(x)`: Función Sphere (De Jong's function 1). Mínimo global: f(0,...,0) = 0
- `rosenbrock(x)`: Función Rosenbrock (De Jong's function 2). Mínimo global: f(1,...,1) = 0
- `rastrigin(x)`: Función Rastrigin. Mínimo global: f(0,...,0) = 0
- `schwefel(x)`: Función Schwefel. Mínimo global: f(420.9687,...,420.9687) = 0
- `griewank(x)`: Función Griewank. Mínimo global: f(0,...,0) = 0
- `ackley(x)`: Función Ackley. Mínimo global: f(0,...,0) = 0
- `levy(x)`: Función Levy. Mínimo global: f(1,...,1) = 0
- `michalewicz(x)`: Función Michalewicz. Problema con múltiples mínimos locales.

#### `get_function()`

```python
def get_function(name)
```

Obtiene una función de prueba por su nombre.

**Parámetros:**
- `name` (str): Nombre de la función de prueba

**Retorna:**
- `callable`: Función de prueba

### Discrete

Problemas de optimización discretos.

#### `knapsack_problem()`

```python
def knapsack_problem(values, weights, capacity)
```

Crea una función objetivo para el problema de la mochila.

**Parámetros:**
- `values` (array-like): Valores de los objetos
- `weights` (array-like): Pesos de los objetos
- `capacity` (float): Capacidad de la mochila

**Retorna:**
- `callable`: Función objetivo para el problema de la mochila

#### `max_cut_problem()`

```python
def max_cut_problem(graph_matrix)
```

Crea una función objetivo para el problema Max-Cut.

**Parámetros:**
- `graph_matrix` (ndarray): Matriz de adyacencia del grafo

**Retorna:**
- `callable`: Función objetivo para el problema Max-Cut

#### `bin_packing_problem()`

```python
def bin_packing_problem(item_sizes, bin_capacity, num_bins)
```

Crea una función objetivo para el problema de empaquetado en contenedores.

**Parámetros:**
- `item_sizes` (array-like): Tamaños de los elementos a empaquetar
- `bin_capacity` (float): Capacidad de cada contenedor
- `num_bins` (int): Número máximo de contenedores

**Retorna:**
- `callable`: Función objetivo para el problema de empaquetado

#### `vehicle_routing_problem()`

```python
def vehicle_routing_problem(distance_matrix, demands, vehicle_capacity)
```

Crea una función objetivo para el problema de enrutamiento de vehículos.

**Parámetros:**
- `distance_matrix` (ndarray): Matriz de distancias entre nodos
- `demands` (array-like): Demandas de cada cliente (nodo)
- `vehicle_capacity` (float): Capacidad de cada vehículo

**Retorna:**
- `callable`: Función objetivo para el problema de enrutamiento

### Combinatorial

Herramientas para problemas de optimización combinatoria.

#### `tsp_create_cities()`

```python
def tsp_create_cities(num_cities, min_coord=0, max_coord=100, seed=None)
```

Genera ciudades aleatorias para el problema del viajante (TSP).

**Parámetros:**
- `num_cities` (int): Número de ciudades a generar
- `min_coord` (float): Coordenada mínima
- `max_coord` (float): Coordenada máxima
- `seed` (int): Semilla para reproducibilidad

**Retorna:**
- `ndarray`: Array de coordenadas (x, y) para cada ciudad

#### `tsp_distance()`

```python
def tsp_distance(route, cities)
```

Calcula la distancia total de una ruta TSP.

**Parámetros:**
- `route` (array-like): Secuencia de ciudades a visitar (índices)
- `cities` (ndarray): Coordenadas de las ciudades

**Retorna:**
- `float`: Distancia total de la ruta

#### `tsp_create_distance_matrix()`

```python
def tsp_create_distance_matrix(cities)
```

Crea una matriz de distancias entre ciudades.

**Parámetros:**
- `cities` (ndarray): Coordenadas de las ciudades

**Retorna:**
- `ndarray`: Matriz de distancias

#### `tsp_plot_solution()`

```python
def tsp_plot_solution(cities, route, title="Solución TSP")
```

Visualiza una solución al problema del viajante.

**Parámetros:**
- `cities` (ndarray): Coordenadas de las ciudades
- `route` (array-like): Secuencia de ciudades (índices)
- `title` (str): Título del gráfico

---

## Módulo Utils

Este módulo contiene herramientas de visualización y funciones auxiliares.

### Visualization

Herramientas para visualizar resultados y análisis.

#### `plot_evolution()`

```python
def plot_evolution(history)
```

Visualiza la evolución del algoritmo genético.

**Parámetros:**
- `history` (dict): Diccionario con historiales de la ejecución

#### `plot_pareto_front()`

```python
def plot_pareto_front(pareto_fitness, objective_names=None, title="Frente de Pareto")
```

Visualiza el frente de Pareto para problemas multi-objetivo.

**Parámetros:**
- `pareto_fitness` (ndarray): Matriz con los valores de fitness de soluciones no dominadas
- `objective_names` (list): Nombres de los objetivos para las etiquetas
- `title` (str): Título del gráfico

#### `plot_population_diversity()`

```python
def plot_population_diversity(history, title="Diversidad de la Población")
```

Visualiza la diversidad de la población a lo largo de las generaciones.

**Parámetros:**
- `history` (dict): Diccionario con historiales de la ejecución
- `title` (str): Título del gráfico

#### `animate_evolution()`

```python
def animate_evolution(history, interval=200, save_path=None)
```

Crea una animación de la evolución de la población.

**Parámetros:**
- `history` (dict): Diccionario con historiales de la ejecución
- `interval` (int): Intervalo entre frames (ms)
- `save_path` (str): Ruta para guardar la animación (None = no guardar)

**Retorna:**
- Animation: Objeto de animación

#### `plot_island_model()`

```python
def plot_island_model(history, num_islands)
```

Visualiza los resultados del modelo de islas.

**Parámetros:**
- `history` (dict): Diccionario con historiales de la ejecución
- `num_islands` (int): Número de islas

### Helpers

Funciones auxiliares para facilitar el uso de la librería.

#### `set_seed()`

```python
def set_seed(seed)
```

Establece una semilla para reproducibilidad de los resultados.

**Parámetros:**
- `seed` (int): Semilla para los generadores de números aleatorios

#### `estimate_runtime()`

```python
def estimate_runtime(objective_function, gene_length, pop_size=100, num_gens=10)
```

Estima el tiempo de ejecución para el problema dado.

**Parámetros:**
- `objective_function` (callable): Función objetivo
- `gene_length` (int): Longitud del genoma
- `pop_size` (int): Tamaño de población a probar
- `num_gens` (int): Número de generaciones para la estimación

**Retorna:**
- `dict`: Estimaciones de tiempo de ejecución

#### `save_results()`

```python
def save_results(results, filename)
```

Guarda los resultados en un archivo JSON.

**Parámetros:**
- `results` (dict): Resultados a guardar
- `filename` (str): Nombre del archivo

#### `load_results()`

```python
def load_results(filename)
```

Carga resultados desde un archivo JSON.

**Parámetros:**
- `filename` (str): Nombre del archivo

**Retorna:**
- `dict`: Resultados cargados

#### `benchmark_operators()`

```python
def benchmark_operators(objective_function, gene_length, bounds=(0, 1), pop_size=50, num_gens=20)
```

Compara diferentes operadores genéticos.

**Parámetros:**
- `objective_function` (callable): Función objetivo
- `gene_length` (int): Longitud del genoma
- `bounds` (tuple): Límites de los genes
- `pop_size` (int): Tamaño de población
- `num_gens` (int): Número de generaciones

**Retorna:**
- `dict`: Resultados comparativos

---

## Ejemplos de Uso

### Optimización Básica

```python
from genetic_algorithm import run_genetic_algorithm
import numpy as np

# Función objetivo: Minimizar la función de Rastrigin
def rastrigin(x):
    n = len(x)
    return -(10 * n + sum(x**2 - 10 * np.cos(2 * np.pi * x)))

# Ejecutar algoritmo genético
result = run_genetic_algorithm(
    objective_function=rastrigin,
    gene_length=5,
    bounds=(-5.12, 5.12),
    pop_size=100,
    num_generations=200,
    adaptive=True
)

print(f"Mejor solución: {result['best_individual']}")
print(f"Mejor fitness: {result['best_fitness']}")
```

### Problema del Viajante (TSP)

```python
from genetic_algorithm import run_genetic_algorithm
from genetic_algorithm.problems.combinatorial import tsp_create_cities, tsp_distance, tsp_plot_solution
import numpy as np

# Crear ciudades aleatorias
num_cities = 20
cities = tsp_create_cities(num_cities, seed=42)

# Función objetivo: Minimizar distancia total
def tsp_objective(route):
    return -tsp_distance(route, cities)  # Negativo para maximización

# Ejecutar algoritmo genético
result = run_genetic_algorithm(
    objective_function=tsp_objective,
    gene_length=num_cities,
    bounds=(0, num_cities-1),
    pop_size=100,
    num_generations=500,
    selection_type="tournament",
    crossover_type="pmx",
    mutation_type="swap",
    encoding="permutation"
)

# Visualizar solución
best_route = result['best_individual'].astype(int)
tsp_plot_solution(cities, best_route, "Mejor Ruta Encontrada")
```

### Optimización con Restricciones

```python
from genetic_algorithm.core.fitness import constrained_fitness
from genetic_algorithm import run_genetic_algorithm

# Problema con restricciones: Optimizar x^2 + y^2 sujeto a x + y <= 1
def objective(x):
    return -(x[0]**2 + x[1]**2)  # Minimizar x^2 + y^2

# Restricción: x + y <= 1 (debe ser >= 0 para ser válida)
def constraint(x):
    return 1 - (x[0] + x[1])

# Función objetivo con restricción
def constrained_objective(x):
    return constrained_fitness(x, objective, [constraint])

# Ejecutar algoritmo genético
result = run_genetic_algorithm(
    objective_function=constrained_objective,
    gene_length=2,
    bounds=(-10, 10),
    pop_size=50,
    num_generations=100
)
```

---

<div align="center">
  <p>Para más información, consulte nuestra documentación completa en <a href="https://github.com/Zaxazgames1/genetic-algorithm-library/wiki">GitHub Wiki</a>.</p>
</div>