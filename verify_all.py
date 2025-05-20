# Guarda como verify_all.py en la carpeta raíz

import numpy as np
import time
from genetic_algorithm import run_genetic_algorithm, run_multi_objective_ga, plot_evolution

def test_simple_optimization():
    """Prueba básica de optimización"""
    print("\n=== PRUEBA DE OPTIMIZACIÓN SIMPLE ===")
    
    # Función sencilla para minimizar (convertida a maximización)
    def objective(x):
        return -(x[0]**2 + x[1]**2)
    
    start_time = time.time()
    result = run_genetic_algorithm(
        objective_function=objective,
        gene_length=2,
        bounds=(-5, 5),
        pop_size=30,
        num_generations=20,
        verbose=True
    )
    elapsed = time.time() - start_time
    
    best_solution = result['best_individual']
    best_fitness = result['best_fitness']
    
    print(f"Tiempo de ejecución: {elapsed:.2f} segundos")
    print(f"Mejor solución encontrada: {best_solution}")
    print(f"Mejor fitness: {best_fitness}")
    
    # Verificación: solución debe estar cerca de [0, 0]
    distance_to_optimum = np.sqrt(np.sum(best_solution**2))
    print(f"Distancia al óptimo conocido: {distance_to_optimum:.6f}")
    
    if distance_to_optimum < 0.5:
        print("✅ Prueba pasada: Solución cerca del óptimo")
    else:
        print("❌ Prueba fallida: Solución lejos del óptimo")
    
    return distance_to_optimum < 0.5

def test_multi_objective():
    """Prueba de optimización multi-objetivo"""
    print("\n=== PRUEBA DE OPTIMIZACIÓN MULTI-OBJETIVO ===")
    
    # Dos objetivos contrapuestos
    def obj1(x):
        return -np.sum(x**2)  # Minimizar suma de cuadrados
    
    def obj2(x):
        return -np.sum((x-2)**2)  # Minimizar distancia a [2,2]
    
    start_time = time.time()
    result = run_multi_objective_ga(
        objective_functions=[obj1, obj2],
        gene_length=2,
        bounds=(-5, 5),
        pop_size=30,
        num_generations=20,
        verbose=True
    )
    elapsed = time.time() - start_time
    
    num_solutions = len(result['pareto_front'])
    
    print(f"Tiempo de ejecución: {elapsed:.2f} segundos")
    print(f"Número de soluciones en el frente de Pareto: {num_solutions}")
    
    # Verificación: debería haber múltiples soluciones
    if num_solutions > 3:
        print("✅ Prueba pasada: Múltiples soluciones en el frente de Pareto")
    else:
        print("❌ Prueba fallida: Pocas soluciones en el frente de Pareto")
    
    return num_solutions > 3

def main():
    """Ejecuta todas las pruebas de verificación"""
    print("INICIANDO VERIFICACIÓN COMPLETA DE LA LIBRERÍA")
    print("=============================================")
    
    test_results = []
    
    # Prueba 1: Optimización simple
    test_results.append(test_simple_optimization())
    
    # Prueba 2: Optimización multi-objetivo
    test_results.append(test_multi_objective())
    
    # Resumen final
    print("\n=== RESUMEN DE VERIFICACIÓN ===")
    if all(test_results):
        print("✅ TODAS LAS PRUEBAS PASARON - La librería funciona correctamente")
    else:
        print("❌ ALGUNAS PRUEBAS FALLARON - Revisar los detalles arriba")
    
    print(f"Pruebas pasadas: {sum(test_results)}/{len(test_results)}")

if __name__ == "__main__":
    main()