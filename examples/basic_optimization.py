import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm import run_genetic_algorithm, plot_evolution

"""
Ejemplo básico de optimización con algoritmos genéticos.
Este script optimiza una función simple para demostrar el uso de la librería.
"""

def main():
    # Definir una función objetivo (maximizar)
    def objective_function(x):
        # Función de Rastrigin negativa (problema de maximización)
        return -(20 + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x))
    
    # Parámetros del algoritmo genético
    gene_length = 2       # Dimensión del problema
    bounds = (-5.12, 5.12)  # Límites de búsqueda
    pop_size = 100        # Tamaño de población
    num_generations = 100 # Número de generaciones
    
    # Ejecutar el algoritmo genético
    print("Ejecutando algoritmo genético...")
    result = run_genetic_algorithm(
        objective_function=objective_function,
        gene_length=gene_length,
        bounds=bounds,
        pop_size=pop_size,
        num_generations=num_generations,
        selection_type="tournament",
        crossover_type="blend",
        mutation_type="gaussian",
        adaptive=True,
        verbose=True
    )
    
    # Mostrar resultados
    print("\nResultados:")
    print(f"Mejor solución encontrada: {result['best_individual']}")
    print(f"Mejor fitness: {result['best_fitness']}")
    
    # Calcular el valor real conocido
    optimal_solution = np.zeros(gene_length)
    optimal_value = objective_function(optimal_solution)
    print(f"Solución óptima conocida: {optimal_solution}")
    print(f"Valor óptimo conocido: {optimal_value}")
    
    # Calcular error
    error = np.linalg.norm(result['best_individual'] - optimal_solution)
    print(f"Error euclídeo: {error}")
    
    # Visualizar la evolución
    plot_evolution(result['history'])
    
    # Visualizar el paisaje de fitness y la solución encontrada
    if gene_length == 2:
        visualize_fitness_landscape(objective_function, bounds, result['best_individual'])

def visualize_fitness_landscape(objective_function, bounds, best_solution):
    """
    Visualiza el paisaje de fitness y la mejor solución encontrada.
    """
    min_val, max_val = bounds
    
    # Crear malla para visualización
    x = np.linspace(min_val, max_val, 100)
    y = np.linspace(min_val, max_val, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calcular valores de fitness para cada punto
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = objective_function([X[i, j], Y[i, j]])
    
    # Crear gráfico 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Superficie
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    # Mejor solución encontrada
    ax.scatter(best_solution[0], best_solution[1], 
              objective_function(best_solution), 
              color='red', s=100, marker='*', label='Mejor solución')
    
    # Óptimo conocido
    ax.scatter(0, 0, objective_function([0, 0]), 
              color='green', s=100, marker='o', label='Óptimo global')
    
    # Etiquetas
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Fitness')
    ax.set_title('Paisaje de Fitness y Solución Encontrada')
    ax.legend()
    
    # Barra de color
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()