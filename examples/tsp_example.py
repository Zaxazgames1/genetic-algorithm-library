import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm import run_genetic_algorithm
from genetic_algorithm.core.crossover import crossover_permutation
from genetic_algorithm.core.mutation import mutation

"""
Ejemplo de optimización para el Problema del Viajante (TSP).
Optimizado para ejecución más rápida manteniendo 500 generaciones.
"""

def main():
    # Definir ciudades (coordenadas x, y)
    cities = np.array([
        [60, 200], [180, 200], [80, 180], [140, 180], [20, 160],
        [100, 160], [200, 160], [140, 140], [40, 120], [100, 120],
        [180, 100], [60, 80], [120, 80], [180, 60], [20, 40],
        [100, 40], [200, 40], [20, 20], [60, 20], [160, 20]
    ])
    
    num_cities = len(cities)
    
    # Precalcular matriz de distancias (optimización clave)
    distance_matrix = calculate_distance_matrix(cities)
    
    # Función objetivo optimizada: minimizar la distancia total
    def objective_function(route):
        route = route.astype(np.int32)  # Usar int32 en lugar de int64
        # Calcular distancia usando vectorización
        cities_in_order = np.append(route, route[0])  # Añadir el regreso a la ciudad inicial
        return -np.sum(distance_matrix[cities_in_order[:-1], cities_in_order[1:]])
    
    # Parámetros del algoritmo genético
    gene_length = num_cities
    bounds = (0, num_cities - 1)
    
    # Reducir tamaño de población para acelerar (aún efectivo)
    pop_size = 50
    
    # Mantener 500 generaciones como solicitado
    num_generations = 500
    
    # Configuración para convergencia más rápida
    print("Resolviendo el Problema del Viajante (TSP) - Versión optimizada...")
    
    # Ejecutar con early stopping para terminar si converge antes
    result = run_genetic_algorithm(
        objective_function=objective_function,
        gene_length=gene_length,
        bounds=bounds,
        pop_size=pop_size,
        num_generations=num_generations,
        selection_type="tournament",
        crossover_type="pmx",
        mutation_type="swap",
        mutation_rate=0.05,
        encoding="permutation",
        adaptive=True,
        verbose=True,
        early_stopping=50,  # Parar si no hay mejora en 50 generaciones
        elitism=True        # Asegurar que el mejor siempre se mantiene
    )
    
    # Mostrar resultados
    best_route = result['best_individual'].astype(int)
    best_distance = -result['best_fitness']
    
    print("\nResultados:")
    print(f"Mejor ruta encontrada en {result['generations']} generaciones")
    print(f"Distancia total: {best_distance:.2f}")
    
    # Visualizar la ruta
    visualize_route(cities, best_route)
    
    # Visualizar la evolución
    from genetic_algorithm import plot_evolution
    plot_evolution(result['history'])

def calculate_distance_matrix(cities):
    """
    Calcula la matriz de distancias entre ciudades de forma optimizada.
    """
    num_cities = len(cities)
    # Usar memoria contigua y tipo de datos eficiente
    distance_matrix = np.zeros((num_cities, num_cities), dtype=np.float32)
    
    # Calcular todas las distancias de una vez con broadcasting
    # Esta es mucho más rápida que los bucles anidados
    for i in range(num_cities):
        # Calcular distancia euclidiana vectorizada
        distance_matrix[i] = np.sqrt(np.sum((cities[i] - cities)**2, axis=1))
    
    return distance_matrix

def visualize_route(cities, route):
    """
    Visualiza la ruta del TSP.
    """
    plt.figure(figsize=(10, 8))
    
    # Dibujar ciudades
    plt.scatter(cities[:, 0], cities[:, 1], c='blue', s=100)
    
    # Numerar ciudades
    for i, (x, y) in enumerate(cities):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
    
    # Dibujar ruta - Optimizado usando vectorización
    route_closed = np.append(route, route[0])  # Añadir regreso al inicio
    for i in range(len(route_closed) - 1):
        city1 = route_closed[i]
        city2 = route_closed[i+1]
        plt.plot([cities[city1, 0], cities[city2, 0]], 
                [cities[city1, 1], cities[city2, 1]], 'r-')
    
    # Resaltar la ciudad inicial/final
    plt.scatter(cities[route[0], 0], cities[route[0], 1], 
               c='green', s=200, alpha=0.5, label='Inicio/Fin')
    
    plt.title('Ruta del Viajante (TSP)')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()