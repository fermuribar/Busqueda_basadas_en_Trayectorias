import numpy as np
import config as conf
import matplotlib.pyplot as plt


#  _               _                      __ _      _                    
# | |             | |                    / _(_)    | |                   
# | |     ___  ___| |_ _   _ _ __ __ _  | |_ _  ___| |__   ___ _ __ ___  
# | |    / _ \/ __| __| | | | '__/ _` | |  _| |/ __| '_ \ / _ \ '__/ _ \ 
# | |___|  __/ (__| |_| |_| | | | (_| | | | | | (__| | | |  __/ | | (_) |
# |______\___|\___|\__|\__,_|_|  \__,_| |_| |_|\___|_| |_|\___|_|  \___/ 

# Función para leer el archivo y procesar los datos
def procesar_archivo(ruta_archivo):
    with open(ruta_archivo, 'r') as archivo:
        lineas = archivo.readlines()
        
        # Tamaño de la matriz (segunda línea del archivo)
        tamaño = int(lineas[1].strip())
        matriz = np.zeros((tamaño, tamaño))
        
        # Rellenar la diagonal principal
        diagonal_principal = [int(x) for x in lineas[2].split()]
        np.fill_diagonal(matriz, diagonal_principal)
        
        # Rellenar el triángulo superior
        fila_actual = 3
        for i in range(tamaño):
            if fila_actual >= len(lineas) or lineas[fila_actual].strip() == '':
                    break
            valores = [int(x) for x in lineas[fila_actual].split()]
            for j in range(i+1, tamaño):
                if valores:
                    matriz[i, j] = valores.pop(0)
                    matriz[j, i] = matriz[i, j]
            
            fila_actual += 1

        # Buscar el peso y el último vector después de la línea en blanco
        fila_actual += 2
        peso = int(lineas[fila_actual].strip())
        vector_pesos = np.array([int(x) for x in lineas[fila_actual+1].split()])

    return matriz, peso, vector_pesos



#  __  __ ______ _______       _    _ ______ _    _ _____  _____  _____ _______ _____ _____          
# |  \/  |  ____|__   __|/\   | |  | |  ____| |  | |  __ \|_   _|/ ____|__   __|_   _/ ____|   /\    
# | \  / | |__     | |  /  \  | |__| | |__  | |  | | |__) | | | | (___    | |    | || |       /  \   
# | |\/| |  __|    | | / /\ \ |  __  |  __| | |  | |  _  /  | |  \___ \   | |    | || |      / /\ \  
# | |  | | |____   | |/ ____ \| |  | | |____| |__| | | \ \ _| |_ ____) |  | |   _| || |____ / ____ \ 
# |_|  |_|______|  |_/_/    \_\_|  |_|______|\____/|_|  \_\_____|_____/   |_|  |_____\_____/_/    \_\

#Clase Solucion
class Solucion:
    def __init__(self) -> None:
        self.solucion = None
        self.peso = None
        self.beneficio = None


class Vecindarios:
    def __init__(self,solucion) ->np.ndarray:
        self.solucion_generadora = solucion.copy()
        
        # 1. Índices de los "1"
        indices_1 = np.where(solucion == 1)[0]

        # 2. Índices de los "0"
        indices_0 = np.where(solucion == 0)[0]
        
        a_repetido = np.repeat(indices_1, len(indices_0), axis=0)
        b_tile = np.tile(indices_0, len(indices_1))
        duplas = np.column_stack((a_repetido, b_tile))

        self.permutaciones = duplas.copy()

        self.indice_permutacion = np.random.permutation(np.arange(0, self.permutaciones.shape[0]))

        self.vector_elementos_sin_explorar = np.ones(self.permutaciones.shape[0], dtype=bool)
    
    def siguiente_vecino(self) -> tuple:
        #devuelve el siguiente vecino en el vecindario. Si ya ha sido todos explorados devuelve la misma solucion que generó el vecindario
        siguiente = self.solucion_generadora.copy()
        permutacion = [-1,-1] #permutacion invalida por defecto 
        

        if(np.any(self.vector_elementos_sin_explorar)):
            indice_elegido = np.argmax(self.vector_elementos_sin_explorar)
            permutacion = self.permutaciones[ self.indice_permutacion[indice_elegido] ][:]
            siguiente[ permutacion[0] ] = 0
            siguiente[ permutacion[1] ] = 1

            self.vector_elementos_sin_explorar[indice_elegido] = False

        return siguiente, permutacion


#Clase problema

class Problema:
    def __init__(self, matriz_valor, peso_max, vector_pesos) -> None:

        self.matriz_valor = matriz_valor
        self.peso_max = peso_max
        self.vector_pesos = vector_pesos
    
    #da una solucion inicial aleatoria 
    def solucion_inicial(self) -> Solucion:
        
        solucion = Solucion()
        solucion.peso = self.peso_max
        solucion.solucion = np.zeros(self.vector_pesos.shape[0])
        # Permuta los indices de manera aleatoria
        indices_aleatorios = np.random.permutation(np.arange(0, solucion.solucion.shape[0]))

        for indice in indices_aleatorios:
            if self.vector_pesos[indice] <= solucion.peso:
                solucion.solucion[indice] = 1
                solucion.peso -= self.vector_pesos[indice]

        solucion.beneficio = np.sum(self.matriz_valor[solucion.solucion.astype(bool), :][:, solucion.solucion.astype(bool)])

        return solucion

    #comprueba si una solucion es factible
    def factible(self,solucion) -> bool:
        peso_solucion_explorar = np.sum(self.vector_pesos[solucion.astype(bool)])
        if peso_solucion_explorar <= self.peso_max:
            return True
        return False
    
    #metodo para rellenar la mochila con elementos que aun quepan en ella
    def completar(self, solucion) -> np.array:
        indices_0 = np.where(solucion == 0)[0]
        aleatorios = np.random.permutation(indices_0)
        peso_dispo = self.peso_max - np.sum(self.vector_pesos[solucion.astype(bool)])
        i = 0
        while i < len(aleatorios) and np.any(self.vector_pesos[aleatorios] <= peso_dispo):
            if self.vector_pesos[aleatorios[i]] <= peso_dispo:
                 solucion[aleatorios[i]] = 1
                 peso_dispo -= self.vector_pesos[aleatorios[i]]
            i+=1

        return solucion
    
    #metodo para hacer que una solucion no factible se haga factible (aleatoriamente)
        #al final se llama a completar
    def factibilizar(self, solucion) -> np.array:
        indices_1 = np.where(solucion == 1)[0]
        aleatorios = np.random.permutation(indices_1)
        for indice in aleatorios:
            solucion[indice] = 0
            if self.factible(solucion):
                break
        solucion = self.completar(solucion)
        return solucion
    
    #calcula todos los parametros de la estructura soluciuon
    def calculo_solucion(self, solucion) -> Solucion:
        solucion_calculada = Solucion()
        solucion_calculada.solucion = solucion.copy()
        solucion_calculada.peso = np.sum(self.vector_pesos[solucion.astype(bool)])
        solucion_calculada.beneficio = np.sum(self.matriz_valor[solucion.astype(bool), :][:, solucion.astype(bool)])
        return solucion_calculada
    
    #calcula los parametros "factorizando" de una solucion a partir de una anterior
    def factorizacion(self, solucion_actual, solucion_nueva, permutacion) -> Solucion:
        solucion_calculada = Solucion()
        solucion_calculada.solucion = solucion_nueva.copy()
        solucion_calculada.peso = np.sum(self.vector_pesos[solucion_nueva.astype(bool)])

        solucion_calculada.beneficio = solucion_actual.beneficio
        solucion_calculada.beneficio = solucion_calculada.beneficio - np.sum(self.matriz_valor[permutacion[0], :][solucion_actual.solucion.astype(bool)]) * 2 + self.matriz_valor[permutacion[0], permutacion[0]]
        solucion_calculada.beneficio = solucion_calculada.beneficio + np.sum(self.matriz_valor[permutacion[1], :][solucion_nueva.astype(bool)]) * 2 - self.matriz_valor[permutacion[1], permutacion[1]]
        
        return solucion_calculada
    
    def mutacion_ILS(self, solucion, t) -> np.ndarray:
        n = solucion.shape[0]
        solucion_aux = solucion.copy()
    
        # 1. Elegir una posición aleatoria
        posicion_inicial = np.random.randint(0, n)
        
        # 2. Extraer los siguientes 20 elementos de forma cíclica
        elementos_a_mutar = np.array([solucion_aux[(posicion_inicial + i) % n] for i in range(t)])
        
        # 3. Hacer un random shuffle al segundo vector
        np.random.shuffle(elementos_a_mutar)
        
        # 4. Asignar nuevamente los valores barajados al primer vector
        for i in range(t):
            solucion_aux[(posicion_inicial + i) % n] = elementos_a_mutar[i]

        # 5. factibilizar la solucion
        if not self.factible(solucion_aux):
            solucion_aux = self.factibilizar(solucion_aux)
        
        return solucion_aux



#   ____  _      
#  |  _ \| |     
#  | |_) | |     
#  |  _ <| |     
#  | |_) | |____ 
#  |____/|______|
    
def BL(matriz_valor, peso_max, vector_pesos, limite, solucion_aleatoria = True, solucion_actual = Solucion()) -> tuple:
    prob = Problema(matriz_valor, peso_max, vector_pesos)
    if solucion_aleatoria:
        solucion_actual = prob.solucion_inicial()
        N = 1
    else:
        N = 0
   
    v = Vecindarios(solucion_actual.solucion)
    permutacion = [1] #para que entre al while la primera vez

    while (permutacion[0] != -1) and (N < limite):
        solucion_a_explorar, permutacion = v.siguiente_vecino()
        if prob.factible(solucion_a_explorar) and (permutacion[0] != -1):
            solucion_a_explorar_calc = prob.factorizacion(solucion_actual, solucion_a_explorar, permutacion)
            N += 1
            
            if solucion_a_explorar_calc.beneficio > solucion_actual.beneficio:
                solucion_actual = solucion_a_explorar_calc
                v = Vecindarios(solucion_actual.solucion)
            
    return (solucion_actual, N)

#   ______  _____ 
#  |  ____|/ ____|
#  | |__  | (___  
#  |  __|  \___ \ 
#  | |____ ____) |
#  |______|_____/ 

def ES(matriz_valor, peso_max, vector_pesos, limite, solucion_aleatoria = True, solucion_actual = Solucion(), mejora = False) -> tuple:
    prob = Problema(matriz_valor, peso_max, vector_pesos)
    if solucion_aleatoria:
        solucion_actual = prob.solucion_inicial()
        N = 1
    else:
        N = 0

    T0 = (0.1 * solucion_actual.beneficio) / -np.log(0.3)
    Tf = 10**-3

    while T0 < Tf:
        Tf*0.1

    max_vecinos = 5 * solucion_actual.solucion.shape[0]
    max_exitos = 0.1 * max_vecinos
    M = limite / max_vecinos
    
    B = T0 - Tf / (M * T0 * Tf)
    T = T0
    mejor_solucion = solucion_actual

    v = Vecindarios(solucion_actual.solucion)
    permutacion = [1] #para que entre al while la primera vez
    Nexitos = 1 #para que entre al while la primera vez

    while (Nexitos > 0 or mejora) and (permutacion[0] != -1 or not mejora) and (N < limite):
        Nexitos = 0
        Nvecinos = 0
        while (permutacion[0] != -1) and (N < limite) and Nexitos < max_exitos and Nvecinos < max_vecinos:
            solucion_a_explorar, permutacion = v.siguiente_vecino()
            if prob.factible(solucion_a_explorar) and (permutacion[0] != -1):            
                solucion_a_explorar_calc = prob.factorizacion(solucion_actual, solucion_a_explorar, permutacion)         
                Af = solucion_actual.beneficio - solucion_a_explorar_calc.beneficio
                N += 1
                Nvecinos += 1
                if Af < 0 or np.random.rand() < np.exp( -Af / T):
                    solucion_actual = solucion_a_explorar_calc
                    Nexitos += 1                
                    v = Vecindarios(solucion_actual.solucion)
                    if Af < 0:
                        mejor_solucion = solucion_actual
        T = T / (1 + B * T)

    return (mejor_solucion, N)

#   ____  __  __ ____  
#  |  _ \|  \/  |  _ \ 
#  | |_) | \  / | |_) |
#  |  _ <| |\/| |  _ < 
#  | |_) | |  | | |_) |
#  |____/|_|  |_|____/ 

def BMB(matriz_valor, peso_max, vector_pesos, limite_bmb, limite_bl, Busqueda = "BL", mejora = False) -> tuple:
    mejor_solucion = Solucion()
    mejor_solucion.beneficio = 0
    N=0
    N_ciclos_bmb = 0
    while N_ciclos_bmb < limite_bmb:
        if Busqueda == "BL":
            solucion_actual, n = BL(matriz_valor, peso_max, vector_pesos, limite_bl)
        elif Busqueda == "ES":
            solucion_actual, n = ES(matriz_valor, peso_max, vector_pesos, limite_bl, mejora=mejora)
        else:
            print("Busqueda no valida")
            return (None, None)
        N += n
        if solucion_actual.beneficio > mejor_solucion.beneficio:
            mejor_solucion = solucion_actual
        
        N_ciclos_bmb += 1
    
    return (mejor_solucion, N)

#   _____ _       _____ 
#  |_   _| |     / ____|
#    | | | |    | (___  
#    | | | |     \___  \ 
#   _| |_| |____ ____) |
#  |_____|______|_____/ 


def ILS(matriz_valor, peso_max, vector_pesos, limite_ils, limite_bl, t, Busqueda = "BL", mejora = False) -> tuple:
    prob = Problema(matriz_valor, peso_max, vector_pesos)
    N=0
    if Busqueda == "BL":
        solucion_actual, n = BL(matriz_valor, peso_max, vector_pesos, limite_bl)
    elif Busqueda == "ES":
        solucion_actual, n = ES(matriz_valor, peso_max, vector_pesos, limite_bl, mejora=mejora)
    else:
        print("Busqueda no valida")
        return (None, None)
    N += n
    mejor_solucion = solucion_actual
    
    N_ciclos_ils = 1
    while N_ciclos_ils < limite_ils:
        #mutacion
        solucion_actual = prob.mutacion_ILS(mejor_solucion.solucion, t)
        #calculo de la solucion mutada
        solucion_actual = prob.calculo_solucion(solucion_actual)
        N += 1
        #busqueda local
        if Busqueda == "BL":
            solucion_actual, n = BL(matriz_valor, peso_max, vector_pesos, limite_bl, solucion_aleatoria = False, solucion_actual = solucion_actual)
        elif Busqueda == "ES":
            solucion_actual, n = ES(matriz_valor, peso_max, vector_pesos, limite_bl, solucion_aleatoria = False, solucion_actual = solucion_actual, mejora=mejora)
        else:
            print("Busqueda no valida")
            return (None, None)
        N += n
        if solucion_actual.beneficio > mejor_solucion.beneficio:
            mejor_solucion = solucion_actual
        
        N_ciclos_ils += 1
    
    return (mejor_solucion, N)