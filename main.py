import config as conf
import algoritmosV4 as alg
import algoritmosP3 as alg3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

#  __  __          _____ _   _ 
# |  \/  |   /\   |_   _| \ | |
# | \  / |  /  \    | | |  \| |
# | |\/| | / /\ \   | | | . ` |
# | |  | |/ ____ \ _| |_| |\  |
# |_|  |_/_/    \_\_____|_| \_|

def main():
    inicio_global = time.time()
    if conf.VER_GRAFICA_DE_MEJORA_SOLO_PARA_UN_PROBLEMA:
        ruta_archivo = conf.PROBLEMA_PARA_VER_GRAFICA # Reemplaza con la ruta de tu archivo
        if not os.path.isfile(ruta_archivo):
            print('Fichero: {}      NO ENCONTRADO'.format(conf.PROBLEMA_PARA_VER_GRAFICA))
            return
        matriz_valores, peso_maximo, vector_pesos = alg.procesar_archivo(ruta_archivo)
        np.random.seed(conf.SEMILLA)
        inicio = time.time()
        sol_agg_1, eva_agg_1, _ = alg.agg(matriz_valores, peso_maximo, vector_pesos,cruce=1)
        fin = time.time()
        duracion_agg_1 = fin - inicio
        if conf.MOSTRAR_CADA_SALIDA:
            print(sol_agg_1.solucion.astype(int))
            print("Agg_1 -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_agg_1.peso ,sol_agg_1.beneficio, duracion_agg_1, eva_agg_1))
            print('')

        np.random.seed(conf.SEMILLA)
        inicio = time.time()
        sol_BL, eva_BL = alg3.BL(matriz_valores, peso_maximo, vector_pesos, limite=conf.MAX_EVALUACIONES)
        fin = time.time()
        duracion_BL = fin - inicio
        if conf.MOSTRAR_CADA_SALIDA:
            print(sol_BL.solucion.astype(int))
            print("BL -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_BL.peso ,sol_BL.beneficio, duracion_BL, eva_BL))
            print('')

        np.random.seed(conf.SEMILLA)
        inicio = time.time()
        sol_ES, eva_ES = alg3.ES(matriz_valores, peso_maximo, vector_pesos, limite=conf.MAX_EVALUACIONES)
        fin = time.time()
        duracion_ES = fin - inicio
        if conf.MOSTRAR_CADA_SALIDA:
            print(sol_ES.solucion.astype(int))
            print("ES -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_ES.peso ,sol_ES.beneficio, duracion_ES, eva_ES))
            print('')

        np.random.seed(conf.SEMILLA)
        inicio = time.time()
        sol_BMB, eva_BMB = alg3.BMB(matriz_valores, peso_maximo, vector_pesos, limite_bmb=conf.LIMITE_BMB, limite_bl=conf.LIMITE_BL_BMB, Busqueda = "BL")
        fin = time.time()
        duracion_BMB = fin - inicio
        if conf.MOSTRAR_CADA_SALIDA:
            print(sol_BMB.solucion.astype(int))
            print("BMB -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_BMB.peso ,sol_BMB.beneficio, duracion_BMB, eva_BMB))
            print('')

        np.random.seed(conf.SEMILLA)
        inicio = time.time()
        sol_BMB_ES, eva_BMB_ES = alg3.BMB(matriz_valores, peso_maximo, vector_pesos, limite_bmb=conf.LIMITE_BMB, limite_bl=conf.LIMITE_BL_BMB, Busqueda = "ES")
        fin = time.time()
        duracion_BMB_ES = fin - inicio
        if conf.MOSTRAR_CADA_SALIDA:
            print(sol_BMB_ES.solucion.astype(int))
            print("BMB_ES -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_BMB_ES.peso ,sol_BMB_ES.beneficio, duracion_BMB_ES, eva_BMB_ES))
            print('')

        np.random.seed(conf.SEMILLA)
        inicio = time.time()
        sol_ILS, eva_ILS = alg3.ILS(matriz_valores, peso_maximo, vector_pesos, limite_bmb=conf.LIMITE_BMB, limite_bl=conf.LIMITE_BL_BMB, t=conf.T_MUTACION, Busqueda = "BL")
        fin = time.time()
        duracion_ILS = fin - inicio
        if conf.MOSTRAR_CADA_SALIDA:
            print(sol_BMB.solucion.astype(int))
            print("ILS -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_ILS.peso ,sol_ILS.beneficio, duracion_ILS, eva_ILS))
            print('')

        np.random.seed(conf.SEMILLA)
        inicio = time.time()
        sol_ILS_ES, eva_ILS_ES = alg3.ILS(matriz_valores, peso_maximo, vector_pesos, limite_bmb=conf.LIMITE_BMB, limite_bl=conf.LIMITE_BL_BMB, t=conf.T_MUTACION, Busqueda = "ES")
        fin = time.time()
        duracion_ILS_ES = fin - inicio
        if conf.MOSTRAR_CADA_SALIDA:
            print(sol_BMB.solucion.astype(int))
            print("ILS -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_ILS_ES.peso ,sol_ILS_ES.beneficio, duracion_ILS_ES, eva_ILS_ES))
            print('')

        return

    # Crear un DataFrame vacío con las columnas específicas
    
    columnas = ['Nombre del problema',  'Solucion Greedy', 'Peso Greedy', 'Bondad Greedy', 'Tiempo Greedy', 
                                        'Solucion agg_1', 'Peso agg_1', 'Bondad agg_1', 'Tiempo agg_1', 
                                        'Solucion BL', 'Peso BL', 'Bondad BL', 'Tiempo BL', 
                                        'Solucion ES', 'Peso ES', 'Bondad ES', 'Tiempo ES', 
                                        'Solucion BMB', 'Peso BMB', 'Bondad BMB', 'Tiempo BMB',
                                        'Solucion BMB_ES', 'Peso BMB_ES', 'Bondad BMB_ES', 'Tiempo BMB_ES',
                                        'Solucion ILS', 'Peso ILS', 'Bondad ILS', 'Tiempo ILS',
                                        'Solucion ILS_ES', 'Peso ILS_ES', 'Bondad ILS_ES', 'Tiempo ILS_ES']
    
        
    tabla = pd.DataFrame(columns=columnas)

    # Uso de la función
    Carga=0
    if (conf.SOLO_EJECUTAR_100 and conf.SOLO_EJECUTAR_200) or (conf.SOLO_EJECUTAR_100 and conf.SOLO_EJECUTAR_300) or (conf.SOLO_EJECUTAR_200 and conf.SOLO_EJECUTAR_300):
        print("Error mala configuracion en los parametros SOLO_EJECUTAR_X. Mas de uno activado")
        return
    elif conf.SOLO_EJECUTAR_100:
        k = 1
        p = 2
    elif conf.SOLO_EJECUTAR_200:
        k = 2
        p = 3
    elif conf.SOLO_EJECUTAR_300:
        k = 3
        p = 4
    else:
        k = 1
        p = 4
    for c in range(k,p):
        for b in conf.EJECUTAR_D:
            for i in range(1,11):
                ruta_archivo = "data/jeu_{}_{}_{}.txt".format(c * 100,b * 25,i)  # Reemplaza con la ruta de tu archivo
                if not os.path.isfile(ruta_archivo):
                    continue
                if not conf.MOSTRAR_CADA_SALIDA:
                    os.system(conf.CLEAR)
                    if Carga == 0:
                        print('->')
                        Carga=1
                    elif Carga == 1:
                        print('-->')
                        Carga=2
                    elif Carga == 2:
                        print('---->')
                        Carga=0

                matriz_valores, peso_maximo, vector_pesos = alg.procesar_archivo(ruta_archivo)

                inicio = time.time()
                solucion_greedy = alg.greedy(matriz_valores, peso_maximo, vector_pesos)
                fin = time.time()
                duracion_Greedy = fin - inicio
                if conf.MOSTRAR_CADA_SALIDA:
                    print(solucion_greedy.solucion.astype(int))
                    print("Greedy -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {}".format(ruta_archivo, peso_maximo - solucion_greedy.peso ,solucion_greedy.beneficio, duracion_Greedy))
                    print('')

                np.random.seed(conf.SEMILLA)
                inicio = time.time()
                sol_agg_1, eva_agg_1, _ = alg.agg(matriz_valores, peso_maximo, vector_pesos,cruce=1)
                fin = time.time()
                duracion_agg_1 = fin - inicio
                if conf.MOSTRAR_CADA_SALIDA:
                    print(sol_agg_1.solucion.astype(int))
                    print("Agg_1 -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_agg_1.peso ,sol_agg_1.beneficio, duracion_agg_1, eva_agg_1))
                    print('')

                np.random.seed(conf.SEMILLA)
                inicio = time.time()
                sol_BL, eva_BL = alg3.BL(matriz_valores, peso_maximo, vector_pesos, limite=conf.MAX_EVALUACIONES)
                fin = time.time()
                duracion_BL = fin - inicio
                if conf.MOSTRAR_CADA_SALIDA:
                    print(sol_BL.solucion.astype(int))
                    print("BL -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_BL.peso ,sol_BL.beneficio, duracion_BL, eva_BL))
                    print('')

                np.random.seed(conf.SEMILLA)
                inicio = time.time()
                sol_ES, eva_ES = alg3.ES(matriz_valores, peso_maximo, vector_pesos, limite=conf.MAX_EVALUACIONES)
                fin = time.time()
                duracion_ES = fin - inicio
                if conf.MOSTRAR_CADA_SALIDA:
                    print(sol_ES.solucion.astype(int))
                    print("ES -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_ES.peso ,sol_ES.beneficio, duracion_ES, eva_ES))
                    print('')

                np.random.seed(conf.SEMILLA)
                inicio = time.time()
                sol_BMB, eva_BMB = alg3.BMB(matriz_valores, peso_maximo, vector_pesos, limite_bmb=conf.LIMITE_BMB, limite_bl=conf.LIMITE_BL_BMB, Busqueda = "BL")
                fin = time.time()
                duracion_BMB = fin - inicio
                if conf.MOSTRAR_CADA_SALIDA:
                    print(sol_BMB.solucion.astype(int))
                    print("BMB -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_BMB.peso ,sol_BMB.beneficio, duracion_BMB, eva_BMB))
                    print('')

                np.random.seed(conf.SEMILLA)
                inicio = time.time()
                sol_BMB_ES, eva_BMB_ES = alg3.BMB(matriz_valores, peso_maximo, vector_pesos, limite_bmb=conf.LIMITE_BMB, limite_bl=conf.LIMITE_BL_BMB, Busqueda = "ES")
                fin = time.time()
                duracion_BMB_ES = fin - inicio
                if conf.MOSTRAR_CADA_SALIDA:
                    print(sol_BMB_ES.solucion.astype(int))
                    print("BMB_ES -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_BMB_ES.peso ,sol_BMB_ES.beneficio, duracion_BMB_ES, eva_BMB_ES))
                    print('')

                np.random.seed(conf.SEMILLA)
                inicio = time.time()
                sol_ILS, eva_ILS = alg3.ILS(matriz_valores, peso_maximo, vector_pesos, limite_ils=conf.LIMITE_ILS, limite_bl=conf.LIMITE_BL_ILS, t=conf.T_MUTACION, Busqueda = "BL")
                fin = time.time()
                duracion_ILS = fin - inicio
                if conf.MOSTRAR_CADA_SALIDA:
                    print(sol_ILS.solucion.astype(int))
                    print("ILS -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_ILS.peso ,sol_ILS.beneficio, duracion_ILS, eva_ILS))
                    print('')

                np.random.seed(conf.SEMILLA)
                inicio = time.time()
                sol_ILS_ES, eva_ILS_ES = alg3.ILS(matriz_valores, peso_maximo, vector_pesos, limite_ils=conf.LIMITE_ILS, limite_bl=conf.LIMITE_BL_ILS, t=conf.T_MUTACION, Busqueda = "ES")
                fin = time.time()
                duracion_ILS_ES = fin - inicio
                if conf.MOSTRAR_CADA_SALIDA:
                    print(sol_ILS_ES.solucion.astype(int))
                    print("ILS_ES -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_ILS_ES.peso ,sol_ILS_ES.beneficio, duracion_ILS_ES, eva_ILS_ES))
                    print('')
                
                tabla.loc[len(tabla)] = [ruta_archivo,  solucion_greedy.solucion, solucion_greedy.peso, solucion_greedy.beneficio, duracion_Greedy, 
                                                        sol_agg_1.solucion, sol_agg_1.peso, sol_agg_1.beneficio, duracion_agg_1, 
                                                        sol_BL.solucion, sol_BL.peso, sol_BL.beneficio, duracion_BL, 
                                                        sol_ES.solucion, sol_ES.peso, sol_ES.beneficio, duracion_ES,
                                                        sol_BMB.solucion, sol_BMB.peso, sol_BMB.beneficio, duracion_BMB,
                                                        sol_BMB_ES.solucion, sol_BMB_ES.peso, sol_BMB_ES.beneficio, duracion_BMB_ES, 
                                                        sol_ILS.solucion, sol_ILS.peso, sol_ILS.beneficio, duracion_ILS, 
                                                        sol_ILS_ES.solucion, sol_ILS_ES.peso, sol_ILS_ES.beneficio, duracion_ILS_ES]
                print('')
                print('')
                print('')
    
    if not conf.MOSTRAR_CADA_SALIDA:
        os.system(conf.CLEAR)
        print('-----------------------------------FIN----------------------------------------')

    fin_global = time.time()
    duracion_global = fin_global - inicio_global

    print("todo las ejecucuciones han tardado: {}".format(duracion_global))


#  _______    _     _              _____                                           _             
# |__   __|  | |   | |            / ____|                                         (_)            
#    | | __ _| |__ | | __ _ ___  | |     ___  _ __ ___  _ __   __ _ _ __ __ _  ___ _  ___  _ __  
#    | |/ _` | '_ \| |/ _` / __| | |    / _ \| '_ ` _ \| '_ \ / _` | '__/ _` |/ __| |/ _ \| '_ \ 
#    | | (_| | |_) | | (_| \__ \ | |___| (_) | | | | | | |_) | (_| | | | (_| | (__| | (_) | | | |
#    |_|\__,_|_.__/|_|\__,_|___/  \_____\___/|_| |_| |_| .__/ \__,_|_|  \__,_|\___|_|\___/|_| |_|
#                                                      | |                                       
#                                                      |_|                                       

    # Paso 1: Extraer 'Tamaño' y 'Densidad' del nombre del problema utilizando expresiones regulares
    # El primer grupo captura secuencias de dígitos (\d+) que siguen al patrón "jeu_" y antes de "_"
    # El segundo grupo captura secuencias de dígitos (\d+) que están entre dos "_"
    tabla['Tamaño'] = tabla['Nombre del problema'].str.extract(r'jeu_(\d+)_').astype(int)
    tabla['Densidad'] = tabla['Nombre del problema'].str.extract(r'jeu_\d+_(\d+)_').astype(int)


    resultados_Greedy = tabla.groupby(['Tamaño', 'Densidad'])[['Bondad Greedy', 'Tiempo Greedy']].mean().reset_index()
    resultados_Greedy['Bondad Greedy'] = resultados_Greedy['Bondad Greedy'].round(2)

    resultados_agg_1 = tabla.groupby(['Tamaño', 'Densidad'])[['Bondad agg_1', 'Tiempo agg_1']].mean().reset_index()
    resultados_agg_1['Bondad agg_1'] = resultados_agg_1['Bondad agg_1'].round(2)

    resultados_BL = tabla.groupby(['Tamaño', 'Densidad'])[['Bondad BL', 'Tiempo BL']].mean().reset_index()
    resultados_BL['Bondad BL'] = resultados_BL['Bondad BL'].round(2)  

    resultados_ES = tabla.groupby(['Tamaño', 'Densidad'])[['Bondad ES', 'Tiempo ES']].mean().reset_index()
    resultados_ES['Bondad ES'] = resultados_ES['Bondad ES'].round(2)

    resultados_BMB = tabla.groupby(['Tamaño', 'Densidad'])[['Bondad BMB', 'Tiempo BMB']].mean().reset_index()
    resultados_BMB['Bondad BMB'] = resultados_BMB['Bondad BMB'].round(2)

    resultados_BMB_ES = tabla.groupby(['Tamaño', 'Densidad'])[['Bondad BMB_ES', 'Tiempo BMB_ES']].mean().reset_index()
    resultados_BMB_ES['Bondad BMB_ES'] = resultados_BMB_ES['Bondad BMB_ES'].round(2)

    resultados_ILS = tabla.groupby(['Tamaño', 'Densidad'])[['Bondad ILS', 'Tiempo ILS']].mean().reset_index()
    resultados_ILS['Bondad ILS'] = resultados_ILS['Bondad ILS'].round(2)

    resultados_ILS_ES = tabla.groupby(['Tamaño', 'Densidad'])[['Bondad ILS_ES', 'Tiempo ILS_ES']].mean().reset_index()
    resultados_ILS_ES['Bondad ILS_ES'] = resultados_ILS_ES['Bondad ILS_ES'].round(2)

    print('')
    print('[*][·][]Tabla Resultados para Greedy')
    print(resultados_Greedy)
    print('')
    print('[*][·][]Tabla Resultados para agg_1')
    print(resultados_agg_1)
    print('')
    print('[*][·][]Tabla Resultados para BL')
    print(resultados_BL)
    print('')
    print('[*][·][]Tabla Resultados para ES')
    print(resultados_ES)
    print('')
    print('[*][·][]Tabla Resultados para BMB')
    print(resultados_BMB)
    print('')
    print('[*][·][]Tabla Resultados para BMB_ES')
    print(resultados_BMB_ES)
    print('')
    print('[*][·][]Tabla Resultados para ILS')
    print(resultados_ILS)
    print('')
    print('[*][·][]Tabla Resultados para ILS_ES')
    print(resultados_ILS_ES)
    print('')
    

    for i in range(k,p):
        #-------------------
        # Filtrar los DataFrames por el tamaño de interés (en este caso, 100)
        resultados_Greedy_c = resultados_Greedy[resultados_Greedy['Tamaño'] == i*100]
        resultados_agg_1_c = resultados_agg_1[resultados_agg_1['Tamaño'] == i*100]
        resultados_BL_c = resultados_BL[resultados_BL['Tamaño'] == i*100]
        resultados_ES_c = resultados_ES[resultados_ES['Tamaño'] == i*100]
        resultados_BMB_c = resultados_BMB[resultados_BMB['Tamaño'] == i*100]
        resultados_BMB_ES_c = resultados_BMB_ES[resultados_BMB_ES['Tamaño'] == i*100]
        resultados_ILS_c = resultados_ILS[resultados_ILS['Tamaño'] == i*100]
        resultados_ILS_ES_c = resultados_ILS_ES[resultados_ILS_ES['Tamaño'] == i*100]
        

        # Calcular las medias de Fitness  para cada algoritmo en tamaño 100
        media_Greedy_c = resultados_Greedy_c['Bondad Greedy'].mean().round(2)
        media_agg_1_c = resultados_agg_1_c['Bondad agg_1'].mean().round(2)
        media_BL_c = resultados_BL_c['Bondad BL'].mean().round(2)
        media_ES_c = resultados_ES_c['Bondad ES'].mean().round(2)
        media_BMB_c = resultados_BMB_c['Bondad BMB'].mean().round(2)
        media_BMB_ES_c = resultados_BMB_ES_c['Bondad BMB_ES'].mean().round(2)
        media_ILS_c = resultados_ILS_c['Bondad ILS'].mean().round(2)
        media_ILS_ES_c = resultados_ILS_ES_c['Bondad ILS_ES'].mean().round(2)

        # La media de Tiempo no se redondea, ya que solo queremos redondear la Bondad
        media_tiempo_Greedy_c = resultados_Greedy_c['Tiempo Greedy'].mean()
        media_tiempo_agg_1_c = resultados_agg_1_c['Tiempo agg_1'].mean()
        media_tiempo_BL_c = resultados_BL_c['Tiempo BL'].mean()
        media_tiempo_ES_c = resultados_ES_c['Tiempo ES'].mean()
        media_tiempo_BMB_c = resultados_BMB_c['Tiempo BMB'].mean()
        media_tiempo_BMB_ES_c = resultados_BMB_ES_c['Tiempo BMB_ES'].mean()
        media_tiempo_ILS_c = resultados_ILS_c['Tiempo ILS'].mean()
        media_tiempo_ILS_ES_c = resultados_ILS_ES_c['Tiempo ILS_ES'].mean()

        
        # Crear un nuevo DataFrame para mostrar los resultados
        resultados_globales_c = pd.DataFrame({
            'Algoritmo': ['Greedy', 'Agg_1', 'BL', 'ES', 'BMB', 'BMB_ES', 'ILS', 'ILS_ES'],
            'Fitness': [ media_Greedy_c, media_agg_1_c, media_BL_c, media_ES_c, media_BMB_c, media_BMB_ES_c, media_ILS_c, media_ILS_ES_c],
            'Tiempo': [media_tiempo_Greedy_c, media_tiempo_agg_1_c, media_tiempo_BL_c, media_tiempo_ES_c, media_tiempo_BMB_c, media_tiempo_BMB_ES_c, media_tiempo_ILS_c, media_tiempo_ILS_ES_c]
        })

        # Si es necesario, ordena la tabla por la columna 'Fitness'
        resultados_globales_c = resultados_globales_c.sort_values(by='Fitness', ascending=False).reset_index(drop=True)

        print('')
        print('[*][·][]Tabla Resultados para Tamaño = {}'.format(100*i))
        # Mostrar la tabla de resultados globales para tamaño 100
        print(resultados_globales_c)

    
    
    if conf.HACER_TXT_DE_TABLA_EJECUCION:
        print('[*][·][]Tabla con todos los problemas')
        print(tabla)
        tabla.to_csv('ejecucion_completa.txt', sep='\t', index=False)



if __name__ == "__main__":
    main()