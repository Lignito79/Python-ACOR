#
# ------------------------------
# Tarea 4: ACOR
# ------------------------------
# José Ángel Rentería Campos
# Matrícula: A00832436
# ------------------------------
#
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class ACOR:
    def __init__(self, m, e, q, k, dimensions, lower_bound, upper_bound, optimize_this):

        self.optimize_this = optimize_this

        self.m = m # NUMERO DE HORMIGAS
        self.e = e # VELOCIDAD DE CONVERGENCIA
        self.q = q # LOCALIDAD DE PROCESO DE BÚSQUEDA
        self.k = k # TAMAÑO DE ARCHIVO
        self.dimensions = dimensions
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.pheromone = np.ones((k, dimensions)) / k
        self.best_solutions = [] # LISTADO DE FEROMONAS POR ITERACIÓN PARA EL PLOTEO
        self.best_solution = float('inf') # INICIALIZADO CON LÍMITE FLOTANTE SUPERIOR INFINITO
        self.best_cost = float('inf') # INICIALIZADO CON LÍMITE FLOTANTE SUPERIOR INFINITO

    def aco_metaheuristic(self, max_iterations):
        iteration = 0
        while iteration < max_iterations:
            self.schedule_activities()
            self.ant_based_solution_construction()
            self.pheromone_update()
            #self.daemon_actions()
            print("--------------------------------------------------")
            print("MEJOR SOLUCION: " + str(self.best_solution))
            print("MEJOR COSTO: " + str(self.best_cost))
            print("--------------------------------------------------")
            self.best_solutions.append((self.best_solution, self.best_cost)) # Esto podría ir en lo de daemon actions? idk
            iteration += 1
        return self.best_solution, self.best_cost

    def schedule_activities(self):
        self.ants = [self.construct_solution() for _ in range(self.m)]

    def ant_based_solution_construction(self):
        for ant in self.ants:
            cost = self.evaluate_solution(ant)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = ant

    def pheromone_update(self):
        self.pheromone *= (1 - self.e)
        for ant in self.ants:
            for i in range(len(ant)):
                self.pheromone[random.randint(0, self.k-1), i] += 1.0 / self.evaluate_solution(ant)

    def construct_solution(self):
        solution = np.zeros(self.dimensions)
        for i in range(self.dimensions):
            probabilities = self.pheromone[:, i] ** self.q
            if probabilities.sum() == 0:
                solution[i] = random.uniform(self.lower_bound, self.upper_bound)
            else:
                #index = np.random.choice(range(self.k), p=probabilities / probabilities.sum())
                solution[i] = random.uniform(self.lower_bound, self.upper_bound)
        return solution

    def evaluate_solution(self, solution):
        return self.optimize_this(solution)

    def plot_contour(self, function, title):
        x = np.linspace(self.lower_bound, self.upper_bound, 100)
        y = np.linspace(self.lower_bound, self.upper_bound, 100)
        X, Y = np.meshgrid(x, y)
        Z = function([X, Y])

        plt.figure()
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        #plt.contourf(X, Y, Z, levels=50, norm=LogNorm())
        plt.colorbar(label='Costo')
        best_x = [solution[0][0] for solution in self.best_solutions]
        best_y = [solution[0][1] for solution in self.best_solutions]
        plt.scatter(best_x, best_y, color='red', label='Mejores soluciones', s=10)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.legend()
        plt.show()

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# AQUÍ TERMINA LA CLASE DE ACOR
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Función de ejemplo por defecto para optimizar
def example_function(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Función Beale
def beale_function(x):
    return (1.5 - x[0] + x[0] * x[1])**2 + \
           (2.25 - x[0] + x[0] * x[1]**2)**2 + \
           (2.625 - x[0] + x[0] * x[1]**3)**2

# Función Booth
def booth_function(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

# Función Branin
def branin_function(x):
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    return a * (x[1] - b * x[0]**2 + c * x[0] - r)**2 + s * (1 - t) * np.cos(x[0]) + s

# Función Goldstein-Price
def goldstein_price_function(x):
    term1 = 1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)
    term2 = 30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2)
    return term1 * term2

# Función Matyas
def matyas_function(x):
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

# Función McCormick
def mccormick_function(x):
    return np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1

# Funcion Three-Hump Camel
def three_hump_camel_function(x):
    return 2 * x[0]**2 - 1.05 * x[0]**4 + (x[0]**6) / 6 + x[0] * x[1] + x[1]**2

# Funcin Six-Hump Camel
def six_hump_camel_function(x):
    return (4 - 2.1 * x[0]**2 + (x[0]**4) / 3) * x[0]**2 + x[0] * x[1] + (-4 + 4 * x[1]**2) * x[1]**2

# Funcion Dixon-Price
def dixon_price_function(x):
    x = np.array(x)
    term1 = (x[0] - 1)**2
    term2 = sum(i * (2 * x[i]**2 - x[i-1])**2 for i in range(1, len(x)))
    return term1 + term2

# Funcion Easom
def easom_function(x):
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi)**2 + (x[1] - np.pi)**2))

function_to_optimize = example_function
lower_limit = -4.5
upper_limit = 4.5

print('''
    Menú de funciones para probar con ACOR:
      1  -. Función Beale
      2  -. Función Booth
      3  -. Función Branin
      4  -. Función Goldstein-Price
      5  -. Función Matyas
      6  -. Función McCormick
      7  -. Funcion Three-Hump Camel
      8  -. Funcin Six-Hump Camel
      9  -. Funcion Dixon-Price
      10 -. Funcion Easom
''')
opcion_function = input("Elija un número del 1 al 10 para seleccionar una función: ")
opcion_function = '#' + opcion_function

# Dependiendo del numero elegido, se usa la correspondiente función
if opcion_function == "#1":
    function_to_optimize = beale_function
    lower_limit = -4.5
    upper_limit = 4.5

elif opcion_function == "#2":
    function_to_optimize = booth_function
    lower_limit = -10
    upper_limit = 10

elif opcion_function == "#3":
    function_to_optimize = branin_function
    lower_limit = -5
    upper_limit = 15

elif opcion_function == "#4":
    function_to_optimize = goldstein_price_function
    lower_limit = -2
    upper_limit = 2

elif opcion_function == "#5":
    function_to_optimize = matyas_function
    lower_limit = -10
    upper_limit = 10

elif opcion_function == "#6":
    function_to_optimize = mccormick_function
    lower_limit = -1.5
    upper_limit = 4

elif opcion_function == "#7":
    function_to_optimize = three_hump_camel_function
    lower_limit = -5
    upper_limit = 5

elif opcion_function == "#8":
    function_to_optimize = six_hump_camel_function
    lower_limit = -3
    upper_limit = 3

elif opcion_function == "#9":
    function_to_optimize = dixon_price_function
    lower_limit = -10
    upper_limit = 10

elif opcion_function == "#10":
    function_to_optimize = easom_function
    lower_limit = -100
    upper_limit = 100

else:
    opcion_function = 'por defecto'
    function_to_optimize = example_function


# Inicializamos el objeto ACOR
acor = ACOR(m = 2,
           e = 0.85, 
           q = 10**4,
           k = 100,
           dimensions = 2,
           lower_bound = lower_limit, 
           upper_bound = upper_limit,
           optimize_this = function_to_optimize)

# Determinamos mejor solución y costo
best_solution, best_cost = acor.aco_metaheuristic(max_iterations=100)

# Ploteo de contorno y mejor solucion por iteración
acor.plot_contour(function_to_optimize, 'Mejor solucion por iteración ' + '(Prueba ' + opcion_function + ')')

print(f"Best solution: {best_solution}")
print(f"Best cost: {best_cost}")
