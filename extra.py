import algoritmosV3 as alg
import numpy as np
import config as conf
import os

ruta_archivo = "data/jeu_{}_{}_{}.txt".format(200, 50, 4)  # Reemplaza con la ruta de tu archivo

matriz_valores, peso_maximo, vector_pesos = alg.procesar_archivo(ruta_archivo)
