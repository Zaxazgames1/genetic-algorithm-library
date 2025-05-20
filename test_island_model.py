# test_island_model.py
import numpy as np
from genetic_algorithm import run_island_model_ga
from genetic_algorithm.utils.visualization import plot_island_model

def sphere_function(x):
    return -sum(x**2)  # Minimizar la suma de cuadrados

result = run_island_model_ga(
    objective_function=sphere_function,
    gene_length=5,
    bounds=(-10, 10),
    num_islands=4,
    pop_size_per_island=30,
    num_generations=50,
    migration_interval=10,
    migration_rate=0.1,
    verbose=True
)

print(f"\nMejor solución global: {result['best_individual']}")
print(f"Mejor fitness global: {result['best_fitness']}")

# Visualizar resultados por isla
plot_island_model(result['history'], 4)