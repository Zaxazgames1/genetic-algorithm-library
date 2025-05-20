@echo off
echo Configurando la librería de Algoritmos Genéticos...

:: Crear estructura de directorios
mkdir genetic_algorithm\core
mkdir genetic_algorithm\utils
mkdir genetic_algorithm\problems
mkdir examples
mkdir docs
mkdir tests

:: Copiar archivos existentes si existen
if exist "functions.py" (
    echo Migrando funciones existentes...
    copy functions.py genetic_algorithm\core\
)

:: Crear archivos básicos
echo Creando estructura de archivos...

:: Crear archivos Python
copy NUL genetic_algorithm\__init__.py
copy NUL genetic_algorithm\core\__init__.py
copy NUL genetic_algorithm\core\population.py
copy NUL genetic_algorithm\core\selection.py
copy NUL genetic_algorithm\core\crossover.py
copy NUL genetic_algorithm\core\mutation.py
copy NUL genetic_algorithm\core\fitness.py
copy NUL genetic_algorithm\utils\__init__.py
copy NUL genetic_algorithm\utils\visualization.py
copy NUL genetic_algorithm\utils\helpers.py
copy NUL genetic_algorithm\problems\__init__.py
copy NUL genetic_algorithm\problems\continuous.py
copy NUL genetic_algorithm\problems\discrete.py
copy NUL genetic_algorithm\problems\combinatorial.py
copy NUL genetic_algorithm\algorithms.py

:: Crear ejemplos
copy NUL examples\basic_optimization.py
copy NUL examples\tsp_example.py
copy NUL examples\multi_objective.py

:: Crear documentación
copy NUL docs\getting_started.md
copy NUL docs\api_reference.md

:: Crear tests
copy NUL tests\test_population.py
copy NUL tests\test_operators.py
copy NUL tests\test_algorithms.py

echo Estructura de proyecto creada correctamente.
echo Para instalar la librería en modo desarrollo, ejecuta: pip install -e .
echo.
echo Proyecto listo para usar. ¡Feliz codificación!