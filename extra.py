import algoritmosV4 as alg4
import algoritmosP3 as alg3
import numpy as np
import config as conf
import time
import os

ruta_archivo = "data/jeu_{}_{}_{}.txt".format(200, 50, 4)  # Reemplaza con la ruta de tu archivo

matriz_valores, peso_maximo, vector_pesos = alg4.procesar_archivo(ruta_archivo)

np.random.seed(conf.SEMILLA)
inicio = time.time()
solucion = alg4.BL_primer_mejor(matriz_valores,peso_maximo,vector_pesos)
fin = time.time()
duracion = fin - inicio
print("BL -> ({}) Con un peso disponible de: {}    y bondad total de: {}|T: {}".format(ruta_archivo, peso_maximo - solucion.peso ,solucion.beneficio,duracion))
print('')

np.random.seed(conf.SEMILLA)
inicio = time.time()
solucion = alg3.BL(matriz_valores,peso_maximo,vector_pesos, limite=conf.MAX_EVALUACIONES)
fin = time.time()
duracion = fin - inicio
print("BL -> ({}) Con un peso disponible de: {}    y bondad total de: {}|T: {}".format(ruta_archivo, peso_maximo - solucion.peso ,solucion.beneficio,duracion))
print('')


np.random.seed(conf.SEMILLA)
inicio = time.time()
solucion = alg3.ES(matriz_valores,peso_maximo,vector_pesos, limite=conf.MAX_EVALUACIONES)
fin = time.time()
duracion = fin - inicio
print("ES -> ({}) Con un peso disponible de: {}    y bondad total de: {}|T: {}".format(ruta_archivo, peso_maximo - solucion.peso ,solucion.beneficio,duracion))
print('')