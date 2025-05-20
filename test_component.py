# test_component.py
from genetic_algorithm.core.population import create_population
from genetic_algorithm.core.crossover import crossover
from genetic_algorithm.core.selection import selection
import numpy as np

# Crear una población
population = create_population(10, 3, -5, 5)
print("Población creada:", population.shape)

# Simular valores de fitness
fitness = -np.sum(population**2, axis=1)
print("Fitness:", fitness)

# Probar selección
parents = selection(population, fitness, 5, "tournament")
print("Padres seleccionados:", parents.shape)

# Probar cruce
offspring = crossover(parents, (8, 3), "blend")
print("Descendencia creada:", offspring.shape)

print("Prueba completada con éxito!")