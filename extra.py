import algoritmosV4 as alg4
import algoritmosP3 as alg3
import numpy as np
import config as conf
import time
import os

ruta_archivo = "data/jeu_{}_{}_{}.txt".format(300, 50, 10)  # Reemplaza con la ruta de tu archivo

matriz_valores, peso_maximo, vector_pesos = alg4.procesar_archivo(ruta_archivo)

np.random.seed(conf.SEMILLA)
inicio = time.time()
solucion, N = alg3.BL(matriz_valores,peso_maximo,vector_pesos, limite=conf.MAX_EVALUACIONES)
fin = time.time()
duracion = fin - inicio
print(N)
print(solucion.solucion.astype(int))
print("BL -> ({}) Con un peso disponible de: {}    y bondad total de: {}|T: {}".format(ruta_archivo, peso_maximo - solucion.peso ,solucion.beneficio,duracion))
print('')


np.random.seed(conf.SEMILLA)
inicio = time.time()
solucion, N = alg3.ES(matriz_valores,peso_maximo,vector_pesos, limite=conf.MAX_EVALUACIONES)
fin = time.time()
duracion = fin - inicio
print(N)
print(solucion.solucion.astype(int))
print("ES -> ({}) Con un peso disponible de: {}    y bondad total de: {}|T: {}".format(ruta_archivo, peso_maximo - solucion.peso ,solucion.beneficio,duracion))
print('')

np.random.seed(conf.SEMILLA)
inicio = time.time()
solucion, N = alg3.BMB(matriz_valores, peso_maximo, vector_pesos, 20, 4500, Busqueda = "BL")
fin = time.time()
duracion = fin - inicio
print(N)
print(solucion.solucion.astype(int))
print("BMB -> ({}) Con un peso disponible de: {}    y bondad total de: {}|T: {}".format(ruta_archivo, peso_maximo - solucion.peso ,solucion.beneficio,duracion))
print('')

np.random.seed(conf.SEMILLA)
inicio = time.time()
solucion, N = alg3.BMB(matriz_valores, peso_maximo, vector_pesos, 20, 4500, Busqueda = "ES")
fin = time.time()
duracion = fin - inicio
print(N)
print(solucion.solucion.astype(int))
print("BMB_ES -> ({}) Con un peso disponible de: {}    y bondad total de: {}|T: {}".format(ruta_archivo, peso_maximo - solucion.peso ,solucion.beneficio,duracion))
print('')

np.random.seed(conf.SEMILLA)
inicio = time.time()
solucion, N = alg3.ILS(matriz_valores, peso_maximo, vector_pesos, 20, 4500, t =20, Busqueda = "BL")
fin = time.time()
duracion = fin - inicio
print(N)
print(solucion.solucion.astype(int))
print("ILS -> ({}) Con un peso disponible de: {}    y bondad total de: {}|T: {}".format(ruta_archivo, peso_maximo - solucion.peso ,solucion.beneficio,duracion))
print('')

np.random.seed(conf.SEMILLA)
inicio = time.time()
solucion, N = alg3.ILS(matriz_valores, peso_maximo, vector_pesos, 20, 4500, t =20, Busqueda = "ES")
fin = time.time()
duracion = fin - inicio
print(N)
print(solucion.solucion.astype(int))
print("ILS_ES -> ({}) Con un peso disponible de: {}    y bondad total de: {}|T: {}".format(ruta_archivo, peso_maximo - solucion.peso ,solucion.beneficio,duracion))
print('')