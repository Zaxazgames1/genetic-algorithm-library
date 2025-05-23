# 🧬 Genetic Algorithm Library

[![PyPI version](https://img.shields.io/badge/pypi-v0.2.0-blue.svg)](https://pypi.org/project/genetic-algorithm-library/)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-wiki-green.svg)](https://github.com/Zaxazgames1/genetic-algorithm-library/wiki)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

<div align="center">
  <img src="https://raw.githubusercontent.com/Zaxazgames1/genetic-algorithm-library/main/docs/images/logo.png" alt="Genetic Algorithm Library Logo" width="300"/>
  <br>
  <em>Optimización evolutiva avanzada para problemas complejos</em>
</div>

## 📋 Descripción General

**Genetic Algorithm Library** es un potente framework de computación evolutiva diseñado para resolver problemas de optimización complejos. Esta librería implementa algoritmos genéticos avanzados con capacidades adaptativas, proporcionando herramientas flexibles y eficientes para investigadores y desarrolladores.

La librería destaca por su amplia variedad de operadores genéticos, soporte para optimización multi-objetivo, y mecanismos de adaptación dinámica que mejoran la convergencia y diversidad poblacional.

## ✨ Características Principales

- **🔄 Múltiples codificaciones genéticas**: 
  - Representaciones reales, binarias, enteras y de permutación
  - Adaptación automática al tipo de problema

- **🔧 Operadores genéticos avanzados**:
  - **Selección**: Torneo, Ruleta, Ranking, SUS (Muestreo Universal Estocástico), Boltzmann
  - **Cruce**: Uniforme, Un punto, Dos puntos, Blend, SBX, PMX (para permutaciones)
  - **Mutación**: Gaussiana, Uniforme, Reset, Adaptativa, Swap e Inversión (para permutaciones)

- **📊 Algoritmos especializados**:
  - Algoritmo genético estándar con adaptación dinámica
  - Optimización multi-objetivo (basada en NSGA-II)
  - Modelo de islas con migración controlada
  - Capacidades paralelas para problemas computacionalmente intensivos

- **📈 Herramientas analíticas integradas**:
  - Visualización avanzada de evolución y convergencia
  - Análisis de frentes de Pareto para optimización multi-objetivo
  - Monitoreo de diversidad poblacional
  - Seguimiento de rendimiento y eficiencia

- **⚙️ Alta personalización**:
  - Integración sencilla de funciones objetivo personalizadas
  - Operadores genéticos personalizables
  - Parámetros adaptables durante la ejecución

## 🚀 Instalación

```bash
pip install genetic-algorithm-library
```

## 🏁 Guía Rápida

### Optimización Simple

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

### Optimización Multi-objetivo

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

### Modelo de Islas

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

## 🛠️ Uso Avanzado

La biblioteca permite un control completo de todos los aspectos del algoritmo genético:

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
    encoding="real"  # Otras: "binary", "integer", "permutation"
)

# Función objetivo personalizada
def my_objective(x):
    return np.sin(x[0]) + np.cos(x[1]) + x[2]**2 - x[3] + x[4]

# Iteración manual del algoritmo
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

## 📊 Visualización de Resultados

La librería incluye herramientas para visualizar y analizar los resultados:

<div align="center">
  <table>
    <tr>
      <td align="center"><img src="https://raw.githubusercontent.com/Zaxazgames1/genetic-algorithm-library/main/docs/images/evolution_plot.png" alt="Evolution Plot" width="400"/></td>
      <td align="center"><img src="https://raw.githubusercontent.com/Zaxazgames1/genetic-algorithm-library/main/docs/images/pareto_front.png" alt="Pareto Front" width="400"/></td>
    </tr>
    <tr>
      <td align="center"><em>Evolución del Fitness</em></td>
      <td align="center"><em>Frente de Pareto Multi-objetivo</em></td>
    </tr>
  </table>
</div>

## 🧪 Problemas Implementados

La librería incluye implementaciones para diversos problemas:

- **Optimización de funciones continuas**:
  - Sphere, Rastrigin, Schwefel, Ackley, Rosenbrock, Griewank y más
  - Fácilmente extensible a funciones personalizadas

- **Problemas de permutación**:
  - Problema del Viajante (TSP)
  - Ordenamiento y secuenciación

- **Problemas discretos**:
  - Mochila (Knapsack)
  - Asignación de tareas
  - Max-Cut

- **Optimización multi-objetivo**:
  - Problemas con objetivos en conflicto
  - Visualización de frentes de Pareto

## 📖 Documentación

Para la documentación completa, visite nuestra [Wiki en GitHub](https://github.com/Zaxazgames1/genetic-algorithm-library/wiki).

También puede consultar los ejemplos incluidos:

- [`examples/basic_optimization.py`](https://github.com/Zaxazgames1/genetic-algorithm-library/blob/main/examples/basic_optimization.py): Optimización básica
- [`examples/tsp_example.py`](https://github.com/Zaxazgames1/genetic-algorithm-library/blob/main/examples/tsp_example.py): Problema del Viajante
- [`examples/multi_objective.py`](https://github.com/Zaxazgames1/genetic-algorithm-library/blob/main/examples/multi_objective.py): Optimización multi-objetivo

## 👥 Colaboradores

- **Johan Rojas** - Desarrollo principal y algoritmos avanzados
- **Julian Lara** - Diseño de API y optimización de rendimiento

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 📚 Cómo Citar

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

## 🔗 Enlaces Útiles

- [Repositorio GitHub](https://github.com/Zaxazgames1/genetic-algorithm-library)
- [Registro de Problemas](https://github.com/Zaxazgames1/genetic-algorithm-library/issues)
- [Página PyPI](https://pypi.org/project/genetic-algorithm-library/)
- [Documentación Wiki](https://github.com/Zaxazgames1/genetic-algorithm-library/wiki)

---

<div align="center">
  <p>Desarrollado con ❤️ para la comunidad de investigación en computación evolutiva</p>
</div>