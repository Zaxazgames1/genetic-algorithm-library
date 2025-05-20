import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm import run_multi_objective_ga, plot_pareto_front

"""
Ejemplo de optimización multi-objetivo.
Este script resuelve un problema con dos objetivos contrapuestos.
"""

def main():
    # Definir funciones objetivo para un problema multi-objetivo
    # Problema: Optimizar forma de una viga (minimizar peso y maximizar rigidez)
    
    def objective1(x):
        # Minimizar peso (proporcional al área de sección)
        return -(x[0] * x[1])  # Negativo para convertir a problema de maximización
    
    def objective2(x):
        # Maximizar rigidez (proporcional al momento de inercia)
        return (x[0] * x[1]**3) / 12.0
    
    # Lista de funciones objetivo
    objective_functions = [objective1, objective2]
    
    # Parámetros del algoritmo genético
    gene_length = 2       # Dimensión del problema (ancho y alto de la viga)
    bounds = (1, 10)      # Límites de búsqueda
    pop_size = 100        # Tamaño de población
    num_generations = 100 # Número de generaciones
    
    # Ejecutar el algoritmo genético multi-objetivo
    print("Ejecutando algoritmo genético multi-objetivo...")
    result = run_multi_objective_ga(
        objective_functions=objective_functions,
        gene_length=gene_length,
        bounds=bounds,
        pop_size=pop_size,
        num_generations=num_generations,
        selection_type="tournament",
        crossover_type="blend",
        mutation_type="gaussian",
        encoding="real",
        verbose=True
    )
    
    # Mostrar resultados
    print("\nResultados:")
    print(f"Número de soluciones en el frente de Pareto: {len(result['pareto_front'])}")
    print("\nAlgunas soluciones representativas:")
    
    # Mostrar algunas soluciones representativas
    n_samples = min(5, len(result['pareto_front']))
    sample_indices = np.linspace(0, len(result['pareto_front']) - 1, n_samples, dtype=int)
    
    for i, idx in enumerate(sample_indices):
        solution = result['pareto_front'][idx]
        fitness = result['pareto_fitness'][idx]
        print(f"Solución {i+1}: Dimensiones = {solution}, Peso = {-fitness[0]:.2f}, Rigidez = {fitness[1]:.2f}")
    
    # Visualizar el frente de Pareto
    objective_names = ["Peso (minimizar)", "Rigidez (maximizar)"]
    plot_pareto_front(result['pareto_fitness'], objective_names)
    
    # Visualizar la evolución del frente de Pareto
    visualize_pareto_evolution(result['history'])
    
    # Visualizar las soluciones en el espacio de decisión
    visualize_design_space(result['pareto_front'], result['pareto_fitness'], bounds)

def visualize_pareto_evolution(history):
    """
    Visualiza la evolución del frente de Pareto a lo largo de las generaciones.
    """
    # Seleccionar generaciones a mostrar
    num_generations = len(history['pareto_fitness'])
    generations_to_show = [0, num_generations // 4, num_generations // 2, 
                          3 * num_generations // 4, num_generations - 1]
    
    plt.figure(figsize=(12, 8))
    
    # Colores para cada generación
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    # Graficar frentes de Pareto para cada generación seleccionada
    for i, gen in enumerate(generations_to_show):
        pareto_fitness = history['pareto_fitness'][gen]
        label = f"Generación {gen+1}"
        
        # Convertir primer objetivo a positivo para visualización
        x = -pareto_fitness[:, 0]  # Peso (convertir a positivo)
        y = pareto_fitness[:, 1]   # Rigidez
        
        plt.scatter(x, y, c=colors[i], label=label, alpha=0.7)
    
    plt.xlabel("Peso (minimizar)")
    plt.ylabel("Rigidez (maximizar)")
    plt.title("Evolución del Frente de Pareto")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_design_space(pareto_front, pareto_fitness, bounds):
    """
    Visualiza las soluciones en el espacio de decisión (dimensiones de la viga).
    
    Parámetros:
    -----------
    pareto_front : ndarray
        Conjunto de soluciones no dominadas
    pareto_fitness : ndarray
        Valores de fitness para las soluciones no dominadas
    bounds : tuple
        Límites de las variables (min_val, max_val)
    """
    plt.figure(figsize=(10, 8))
    
    # Convertir fitness para visualización
    weight = -pareto_fitness[:, 0]  # Convertir a valores positivos
    stiffness = pareto_fitness[:, 1]
    
    # Normalizar para tamaño de puntos
    size_min, size_max = 50, 200
    if np.max(stiffness) > np.min(stiffness):
        sizes = size_min + (size_max - size_min) * (stiffness - np.min(stiffness)) / (np.max(stiffness) - np.min(stiffness))
    else:
        sizes = np.full_like(stiffness, (size_min + size_max) / 2)
    
    # Graficar espacio de diseño
    scatter = plt.scatter(
        pareto_front[:, 0],  # Ancho
        pareto_front[:, 1],  # Alto
        c=weight,            # Color según peso
        s=sizes,             # Tamaño según rigidez
        cmap='viridis',
        alpha=0.7
    )
    
    plt.colorbar(scatter, label="Peso")
    
    # Añadir anotaciones para algunas soluciones
    for i in range(min(5, len(pareto_front))):
        plt.annotate(
            f"({pareto_front[i, 0]:.2f}, {pareto_front[i, 1]:.2f})",
            (pareto_front[i, 0], pareto_front[i, 1]),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.xlabel("Ancho de la Viga")
    plt.ylabel("Alto de la Viga")
    plt.title("Espacio de Diseño - Soluciones No Dominadas")
    plt.grid(True)
    
    # Líneas de referencia para secciones cuadradas
    x = np.linspace(bounds[0], bounds[1], 100)
    plt.plot(x, x, 'k--', alpha=0.5, label="Sección Cuadrada")
    
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()