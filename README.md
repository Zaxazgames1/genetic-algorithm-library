# Genetic Algorithm Library

[![PyPI version](https://img.shields.io/badge/pypi-v0.2.0-blue.svg)](https://pypi.org/project/genetic-algorithm-library/)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Genetic Algorithm Library es una librería robusta y completa para implementar algoritmos genéticos y evolutivos en problemas de optimización. Proporciona un framework flexible y potente para resolver problemas complejos mediante técnicas de computación evolutiva avanzadas.

## Características principales

- **Múltiples codificaciones genéticas**: Soporte para representaciones reales, binarias, enteras y de permutación
- **Amplia variedad de operadores genéticos**:
  - **Selección**: Torneo, Ruleta, Ranking, SUS (Muestreo Universal Estocástico), Boltzmann
  - **Cruce**: Uniforme, Un punto, Dos puntos, Blend, SBX, PMX (para permutaciones)
  - **Mutación**: Gaussiana, Uniforme, Reset, Adaptativa, Swap e Inversión (para permutaciones)
- **Adaptabilidad dinámica**: Ajuste automático de parámetros y tasas de mutación durante la ejecución
- **Algoritmos avanzados**:
  - Algoritmo genético estándar
  - Optimización multi-objetivo (basada en NSGA-II)
  - Modelo de islas con migración
- **Herramientas de análisis y visualización**: Gráficos detallados de evolución, frentes de Pareto y diversidad poblacional
- **Alta personalización**: Fácil integración de funciones objetivo y operadores personalizados
- **Documentación detallada**: Ejemplos, guías y documentación completa de la API

## Instalación

```bash
pip install genetic-algorithm-library
```

## Guía rápida

### Optimización simple

```python
from genetic_algorithm import run_genetic_algorithm, plot_evolution
import numpy as np

# Definir función objetivo (maximizar)
def objective_function(x):
    return -(x[0]**2 + x[1]**2)  # Minimizar la suma de cuadrados

# Ejecutar algoritmo genético
result = run_genetic_algorithm(
    objective_function=objective_function,
    gene_length=2,               # 2 parámetros a optimizar
    bounds=(-10, 10),            # Límites de búsqueda
    pop_size=100,                # Tamaño de población
    num_generations=50,          # Número de generaciones
    selection_type="tournament", # Método de selección
    adaptive=True,               # Mutación adaptativa
    verbose=True                 # Mostrar progreso
)

# Mostrar resultados
print(f"Mejor solución: {result['best_individual']}")
print(f"Mejor fitness: {result['best_fitness']}")

# Visualizar evolución
plot_evolution(result['history'])
```

### Optimización multi-objetivo

```python
from genetic_algorithm import run_multi_objective_ga, plot_pareto_front

# Definir funciones objetivo
def objective1(x):
    return -sum(x**2)  # Minimizar (convertido a maximizar con negativo)

def objective2(x):
    return -sum((x-2)**2)  # Minimizar distancia a (2,2,...,2)

# Ejecutar algoritmo genético multi-objetivo
result = run_multi_objective_ga(
    objective_functions=[objective1, objective2],
    gene_length=2,
    bounds=(-5, 5),
    pop_size=200,
    num_generations=100
)

# Visualizar frente de Pareto
plot_pareto_front(
    result['pareto_fitness'],
    objective_names=["Suma de cuadrados", "Distancia a (2,2)"]
)
```

### Modelo de islas

```python
from genetic_algorithm import run_island_model_ga

# Ejecutar algoritmo con modelo de islas
result = run_island_model_ga(
    objective_function=my_objective,
    gene_length=5,
    bounds=(-10, 10),
    num_islands=4,              # Número de islas
    pop_size_per_island=50,     # Población por isla
    num_generations=100,
    migration_interval=10,      # Migración cada 10 generaciones
    migration_rate=0.1          # 10% de individuos migran
)
```

## Uso avanzado

La biblioteca permite un control avanzado de todos los aspectos del algoritmo genético:

```python
import numpy as np
from genetic_algorithm import (
    create_population,
    selection,
    crossover,
    mutation,
    adaptive_mutation,
    fitness_function
)

# Crear población inicial con codificación específica
population = create_population(
    size=50, 
    gene_length=5, 
    min_val=-5, 
    max_val=5,
    encoding="real"  # Otras opciones: "binary", "integer", "permutation"
)

# Función objetivo personalizada
def my_objective(x):
    return np.sin(x[0]) + np.cos(x[1]) + x[2]**2 - x[3] + x[4]

# Iteración manual
for generation in range(100):
    # Evaluar fitness
    fitness_values = np.array([fitness_function(ind, my_objective) for ind in population])
    
    # Seleccionar padres
    parents = selection(
        population, 
        fitness_values, 
        num_parents=25, 
        selection_type="rank"  # Otras: "tournament", "roulette", "sus", "boltzmann"
    )
    
    # Crear descendencia mediante cruce
    offspring = crossover(
        parents, 
        offspring_size=(25, 5), 
        crossover_type="blend"  # Otras: "uniform", "single_point", "two_point", "sbx"
    )
    
    # Aplicar mutación adaptativa basada en fitness
    offspring = adaptive_mutation(
        offspring,
        fitness_values[:25],  # Fitness de los padres
        np.max(fitness_values),
        np.mean(fitness_values),
        min_val=-5,
        max_val=5,
        base_rate=0.05
    )
    
    # Actualizar población con elitismo
    best_idx = np.argmax(fitness_values)
    population = np.vstack([population[best_idx:best_idx+1], parents[:-1], offspring])
```

## Ejemplos incluidos

La biblioteca incluye ejemplos completos para diversos problemas:

- **Optimización de funciones continuas**: minimización de funciones matemáticas
- **Problema del Viajante (TSP)**: optimización de rutas mediante permutaciones
- **Optimización multi-objetivo**: problemas con objetivos en conflicto
- **Modelo de islas**: optimización con múltiples subpoblaciones en paralelo

## Documentación

Para la documentación completa, visite nuestra [Wiki en GitHub](https://github.com/Zaxazgames1/genetic-algorithm-library/wiki).

## Colaboradores

- Julian Lara
- Johan Rojas

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

## Cómo citar

Si utiliza esta biblioteca en su investigación, por favor cite:

```
@software{genetic_algorithm_library,
  author = {Lara, Julian and Rojas, Johan},
  title = {Genetic Algorithm Library},
  url = {https://github.com/Zaxazgames1/genetic-algorithm-library},
  version = {0.2.0},
  year = {2025},
}
```