"""
Herramientas Computacionales: El Arte de la Programación

Evidencia de Proyecto

Equipo: Jonathan Uziel Medina Rodríguez (A01255048), Pablo Ernesto Cruz Moreno (A01255437) y
        Miguel de Jesús Degollado Macías (A01255388).

Docente: Baldomero Olvera Villanueva

Fecha: 23/03/2025

Descripción: Programa que aplica filtros de convolución a una radiografía para detectar cuerpos extraños,
             enfermedades respiratorias, etc.
"""

# Librerías
import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt

# Función para agregar un filtro de convolución a una imagen. Complejidad O(n^2) al recorrer n columnas de n filas.
def convolucion(imagen, filtro, padding_y=1, padding_x=1):
    # Filas y columnas de la imagen.
    filaImg, colImg = imagen.shape

    # Filas y columnas del filtro.
    filaF, colF = filtro.shape

    # Si no se especifica padding, se usa el valor por defecto.
    padding_y = int((filaF - 1) / 2) if padding_y is None else padding_y
    padding_x = int((colF - 1) / 2) if padding_x is None else padding_x

    # Matriz de ceros con el tamaño de la imagen.
    matriz = np.zeros(imagen.shape)

    # Matriz de ceros con el tamaño de la imagen más el padding.
    imagenPadding = np.zeros((filaImg + (padding_y * 2), colImg + (padding_x * 2)))

    # Toda la parte que no abarca el padding será abarcada por la imagen.
    imagenPadding[padding_y:imagenPadding.shape[0] - padding_y,
                  padding_x:imagenPadding.shape[1] - padding_x] = imagen

    # Recorrer n filas de la imagen.
    for i in range(filaImg):
        # Recorrer n columnas de la imagen.
        for j in range(colImg):
            # Operación de convolución en cada celda.
            matriz[i][j] = np.sum(imagenPadding[i:i + filaF, j:j + colF] * filtro)

    # Matriz con filtros aplicados.
    return matriz


"""
Función para guardar la imagen en la carpeta "Fotos". Complejidad O(1) al no tener iteraciones;
solo hay un condicional y llamadas a funciones.
""" 
def guardar_imagen(imagen, nombre_imagen):
    # Checar si existe la carpeta "Fotos".
    if not os.path.exists('Fotos'):
        os.makedirs('Fotos')
    
    # Crear la imagen.
    cv2.imwrite(f'Fotos/{nombre_imagen}.png', imagen)


# Programa principal
if __name__ == "__main__":

    # Ingresar nombre o directorio del archivo.
    archivo = input("\nIngrese el nombre de la imagen: ")

    # Checar si la imagen existe.
    if os.path.isfile(archivo) == False:
        print("\nNo se pudo encontrar la imagen.")
    
    else:
        # Mapas de color admitidos por el programa.
        colorMapArr = ["bone", "afmhot", "gray"]
        
        # Mapa de color ingresado por el usuario.
        colorMap = ""

        # Mientras no se ha ingresado un nombre de mapa de color válido, se le pide que ingrese uno válido. 
        while colorMap not in colorMapArr:
            colorMap = input("\nIngrese el nombre del mapa de color que quiera utilizar:\n"
                            "\t1. bone\n"
                            "\t2. afmhot\n"
                            "\t3. gray\n"
                            "____________________________________________________________\n"
                            "Opción: ")

            if colorMap not in colorMapArr:
                print("\n\nEl mapa de color no es válido. Ingrese uno de los solicitados.\n\n")
                time.sleep(2)
            else:
                break

        # Solicitar al usuario tipo y cantidad de padding.
        padding_y = int(input("\nIngrese la cantidad de padding en el eje Y (por defecto 1): ") or 1)
        padding_x = int(input("\nIngrese la cantidad de padding en el eje X (por defecto 1): ") or 1)

        # Se lee la imagen.
        imagen = cv2.imread(archivo)

        # La imagen se convierte a blanco y negro (escala de grises).
        imagenEG = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)

        # Matrices de filtros.
        bordes = np.array([[-1, 0, -1],
                           [-1, -3, -1],
                           [-1, 0, -1]])

        scharr = np.array([[-3, 10, 3],
                           [1, -5, 1],
                           [-3, 0, 3]])

        sobel = np.array([[-1, -2, -1],
                          [-2, 9, 2],
                          [1, 2, 1]])

        nitidez = np.array([[0, -1, 0],
                            [-1, 9, -1],
                            [0, -1, 0]])

        # Aplicación de los filtros
        mascara1 = convolucion(imagenEG, nitidez, padding_y, padding_x)
        print("\nMáscara #1 aplicada.")
        mascara2 = convolucion(mascara1, scharr, padding_y, padding_x)
        print("\nMáscara #2 aplicada.")
        mascara3 = convolucion(mascara2, bordes, padding_y, padding_x)
        print("\nMáscara #3 aplicada.")
        mascara4 = convolucion(mascara3, sobel, padding_y, padding_x)
        print("\nMáscara #4 aplicada.")

        time.sleep(1)

        # Definir la posición, mapa de color y título de la imagen con escala de grises
        plt.subplot(1, 2, 1)
        plt.imshow(imagenEG, cmap="gray")
        plt.title("Imagen con Escala de Grises")

        # Definir la posición, mapa de color y título de la imagen con el filtro aplicado
        plt.subplot(1, 2, 2)
        plt.imshow(mascara4, cmap=colorMap)
        plt.title("Imagen con el Filtro Aplicado")

        # Mostrar las imágenes
        plt.show()

        # Guardar la imagen procesada
        guardar_imagen(mascara4, "Imagen_Procesada")
        print("\nLa imagen filtrada ha sido guardada en la carpeta 'Fotos'.")


"""
Referencias:

- Choosing Colormaps in Matplotlib. (s. f.). Matplotlib. Recuperado el 22 de marzo de 2025 de https://matplotlib.org/stable/users/explain/colors/colormaps.html
- Función de convolución. (s. f.). ArcGIS Desktop. Recuperado el 22 de marzo de 2025 de https://desktop.arcgis.com/es/arcmap/latest/manage-data/raster-and-images/convolution-function.htm
- GeeksforGeeks. (2020a, 22 abril). Matplotlib.pyplot.imshow() in Python. GeeksforGeeks. Recuperado el 20 de marzo de 2025 de https://www.geeksforgeeks.org/matplotlib-pyplot-imshow-in-python/
- GeeksforGeeks. (2022b, 10 febrero). How to Fix: ValueError: setting an array element with a sequence. GeeksforGeeks. Recuperado el 20 de marzo de 2025 de https://www.geeksforgeeks.org/how-to-fix-valueerror-setting-an-array-element-with-a-sequence/
- GeeksforGeeks. (2023c, 14 marzo). Introduction to Convolutions using Python. GeeksforGeeks. Recuperado el 20 de marzo de 2025 de https://www.geeksforgeeks.org/introduction-to-convolutions-using-python/
- GeeksforGeeks. (2024d, 24 abril). Python - How to Check if a file or directory exists. GeeksforGeeks. Recuperado el 20 de marzo de 2025 de https://www.geeksforgeeks.org/python-check-if-a-file-or-directory-exists/
- GeeksforGeeks. (2024e, 22 julio). Types of convolution kernels. GeeksforGeeks. Recuperado el 20 de marzo de 2025 de https://www.geeksforgeeks.org/types-of-convolution-kernels/
- GeeksforGeeks. (2024f, 2 agosto). Python OpenCV | cv2.imread() method. GeeksforGeeks. Recuperado el 20 de marzo de 2025 de https://www.geeksforgeeks.org/python-opencv-cv2-imread-method/
- GeeksforGeeks. (2024g, 28 agosto). Numpy.sum() in Python. GeeksforGeeks. Recuperado el 21 de marzo de 2025 de https://www.geeksforgeeks.org/numpy-sum-in-python/
- GeeksforGeeks. (2024h, 25 noviembre). How to display multiple images in one figure correctly in Matplotlib? GeeksforGeeks. Recuperado el 20 de marzo de 2025 de https://www.geeksforgeeks.org/how-to-display-multiple-images-in-one-figure-correctly-in-matplotlib/
- GeeksforGeeks. (2025i, 24 enero). Numpy.zeros() in Python. GeeksforGeeks. Recuperado el 20 de marzo de 2025 de https://www.geeksforgeeks.org/numpy-zeros-python/
- 8.2 Matriz de convolución. (s. f.). GIMP. https://docs.gimp.org/2.6/es/plug-in-convmatrix.html
- Oliveres, J., & Escalante, B. (2011). Convolución y Filtrado [Presentación; Digital]. Universidad Nacional Autónoma de México. https://lapi.fi-p.unam.mx/wp-content/uploads/6-Filtros-y-morfologia.pdf 
- Olvera, B. (s. f.-a). Convolución [Diapositivas; Digital]. Tecnológico de Monterrey. https://experiencia21.tec.mx/courses/554652/discussion_topics/3503409
- Olvera, B. (s. f.-b). Procesamiento de Imágenes y Visión Computacional [Diapositivas; Digital]. Tecnológico de Monterrey. https://experiencia21.tec.mx/courses/554652/discussion_topics/3503409
- Ormesher, I. (2021, 14 diciembre). Convolution Filters. Medium. Recuperado el 20 de marzo de 2025 de https://medium.com/@ianormy/convolution-filters-4971820e851f

Basado en los siguientes códigos:

- "convolution.py" por Abhisek Jana. Recuperado de https://github.com/benjaminva/semena-tec-tools-vision/tree/master/Scripts/Ejemplos
- "simple_conv.py" por Abhisek Jana y Benajmin Valdes. Recuperado de https://github.com/benjaminva/semena-tec-tools-vision/tree/master/Scripts/Ejemplos
- "simple_sobel.py" por Abhisek Jana y Benajmin Valdes. Recuperado de https://github.com/benjaminva/semena-tec-tools-vision/tree/master/Scripts/Ejemplos
"""
