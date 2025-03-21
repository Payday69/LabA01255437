# Pablo Ernesto Moreno Cruz (A01255437)
import numpy as np
import cv2
import matplotlib.pyplot as plt

def convolution(image, kernel, average=False, verbose=False):
    
    # Convertir la imagen a escala de grises si tiene más de un canal
    is_color = False
    if len(image.shape) == 3:
        print("Imagen con 3 canales detectada: {}".format(image.shape))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Convertida a escala de grises: {}".format(gray_image.shape))
        is_color = True
    else:
        print("Imagen en escala de grises detectada: {}".format(image.shape))
        gray_image = image
    
    print("Tamaño del kernel: {}".format(kernel.shape))
    
    if verbose and is_color:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Imagen original a color")
        plt.show()
    elif verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Imagen original en escala de grises")
        plt.show()
    
    # Dimensiones de la imagen y del kernel
    image_row, image_col = gray_image.shape
    kernel_row, kernel_col = kernel.shape
    
    # Inicializar la imagen de salida con ceros
    output = np.zeros(gray_image.shape)
    
    # Calcular el padding necesario para mantener el tamaño original de la imagen
    pad_height = kernel_row // 2
    pad_width = kernel_col // 2
    
    # Crear imagen con padding de ceros
    padded_image = np.pad(gray_image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    
    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Imagen con Padding")
        plt.show()
    
    # Aplicar la convolución recorriendo la imagen
    for row in range(image_row):
        for col in range(image_col):
            region = padded_image[row:row + kernel_row, col:col + kernel_col]
            output[row, col] = np.sum(kernel * region)
            if average:
                output[row, col] /= kernel.size
    
    # Normalizar la imagen de salida para mejorar la visualización
    output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
    output = np.uint8(output)
    
    print("Tamaño de la imagen de salida: {}".format(output.shape))
    
    if verbose:
        plt.imshow(output, cmap='magma')
        plt.title("Imagen con Convolución")
        plt.show()
    
    return output

def sobel_edge_detection(image, verbose=False):
   
    # Definir los filtros de Sobel para detección de bordes en X y Y
    sobel_x = np.array([[-1, 69, 1],
                        [-2, 69, 2],
                        [-1, 69, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [69, 69, 69],
                        [1, 2, 1]])
    
    # Aplicar convolución con los filtros de Sobel
    edge_x = convolution(image, sobel_x, verbose)
    edge_y = convolution(image, sobel_y, verbose)
    
    # Calcular la magnitud del gradiente combinando las direcciones X e Y
    gradient_magnitude = np.sqrt(np.square(edge_x) + np.square(edge_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    gradient_magnitude = np.uint8(gradient_magnitude)
    
    if verbose:
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("Detección de Bordes - Sobel")
        plt.show()
    
    return gradient_magnitude

if __name__ == '__main__':
    # Ruta de la imagen de entrada
    image_path = 'CIrculo.jpg'  # Aqui se pone el nombre de la ruta de la imagen
    
    # Cargar la imagen en color
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: No se pudo cargar la imagen.")
    else:
        # Mostrar la imagen original en color
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Imagen Original a Color")
        plt.show()
        
        # Convertir la imagen a escala de grises para el procesamiento
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar padding a la imagen y mostrarla
        pad_size = 69
        padded_image = np.pad(gray_image, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)
        plt.imshow(padded_image, cmap='gray')
        plt.title("Imagen con Padding")
        plt.show()
        
        # Aplicar detección de bordes con Sobel y mostrar el resultado
        sobel_result = sobel_edge_detection(gray_image, verbose=True)
        
        # Guardar la imagen resultante
        cv2.imwrite('resultado_sobel.jpg', sobel_result)
