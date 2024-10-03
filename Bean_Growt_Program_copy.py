# =======================================
#        LIBRERÍAS Y CARGA DE MODELO
# =======================================
import tkinter as tk
from tkinter import Label, filedialog, Button
from PIL import Image, ImageTk, ImageEnhance, ImageDraw, ImageFont
import os
import numpy as np
import tensorflow as tf
from keras.utils import img_to_array
import cv2
import random
from inference_sdk import InferenceHTTPClient

#URL_MODELO = r"C:\Users\paoli\OneDrive\Documentos\Trabajo_Grado\GUI_FINAL"
URL_MODELO = os.getcwd()

# Cargar el modelo previamente entrenado
modelo = tf.keras.models.load_model(fr"{URL_MODELO}\Mejor_Modelo_fold_Entrega_final_TG.keras")

# =======================================
#              FUNCIONES
# =======================================

# Función para cambiar a la segunda pantalla
def ir_a_siguiente():
    for widget in ventana.winfo_children():
        widget.destroy()

    texto_segunda_pantalla = """A continuación ingrese la imagen, tenga en cuenta que para el funcionamiento del programa
    esta imagen debe estar en formato .jpg (el programa convierte a este formato de manera automática) y entre mejor resolución tenga,
    mejor funcionará el programa. Recuerde que el programa está diseñado para fotos de cultivos de frijol tomadas a 15 metros de altura 
    y con luz diurna."""

    label_texto_segunda = tk.Label(ventana, text=texto_segunda_pantalla, wraplength=700, bg='#67b93e', fg='white', font=("Arial Black", 14), anchor="center", justify="left")
    label_texto_segunda.pack(pady=20)

    boton_cargar = Button(ventana, text="Cargar imagen", command=cargar_imagen, font=("Arial", 12), bg="white", fg="black")
    boton_cargar.pack(pady=10)

    global label_mensaje
    label_mensaje = Label(ventana, text="", bg='#67b93e', fg='white', font=("Arial", 12))
    label_mensaje.pack(pady=10)


# Función para cargar imagen y verificar formato
def cargar_imagen():
    global imagen_cargada, ruta_imagen_cargada, imagen_original

    ruta_imagen_cargada = filedialog.askopenfilename(title="Seleccionar imagen", filetypes=[("Imagenes", "*.jpg;*.png;*.jpeg;*.bmp")])

    if ruta_imagen_cargada:
        try:
            imagen_original = Image.open(ruta_imagen_cargada)  # Guardar la imagen original
            nombre_archivo, extension = os.path.splitext(ruta_imagen_cargada)

            # Convertir a JPG si no es JPG
            if extension.lower() != '.jpg':
                imagen_cargada = imagen_original.convert("RGB")
            else:
                imagen_cargada = imagen_original
            
            # Guardar la imagen en formato .jpg (aunque sea en memoria)
            imagen_cargada.save("temp_image.jpg", "JPEG")

            label_mensaje.config(text="Imagen cargada correctamente", fg="lightgreen")

            # Crear el botón de comenzar análisis después de cargar la imagen
            boton_comenzar = Button(ventana, text="Comenzar análisis", command=comenzar_analisis, font=("Arial", 12), bg="white", fg="black")
            boton_comenzar.pack(pady=10)

        except Exception as e:
            label_mensaje.config(text="Error al cargar la imagen: " + str(e), fg="red")
    else:
        label_mensaje.config(text="No se seleccionó ninguna imagen", fg="red")


# Función para comenzar el análisis con el modelo
def comenzar_analisis():
    # Limpiar la ventana y mostrar la pantalla de carga
    for widget in ventana.winfo_children():
        widget.destroy()

    ventana.configure(bg='#67b93e')

    # Cargar la imagen GIF de carga
    ruta_gif = fr"{URL_MODELO}\Imagenes\Reloj_de_Carga.gif"
    gif_carga = Image.open(ruta_gif)

    # Crear una lista para almacenar los frames del GIF
    frames = []
    try:
        while True:
            frames.append(ImageTk.PhotoImage(gif_carga.copy()))
            gif_carga.seek(len(frames))  # Avanzar al siguiente frame
    except EOFError:
        pass  # Termina cuando no hay más frames

    # Función para actualizar el frame del GIF
    def actualizar_gif(indice):
        #if indice > 3: #solo soporta la cantidad
        #    return
        frame = frames[indice]
        label_gif.config(image=frame)
        ventana.after(100, actualizar_gif, (indice + 1) % len(frames))  # Cambiar al siguiente frame

    label_gif = Label(ventana, bg='#67b93e')
    label_gif.pack(pady=20)

    # Iniciar la animación
    actualizar_gif(0)

    # Realizar predicción con el modelo después de un pequeño retraso
    ventana.after(5000, realizar_prediccion)  # Cambiar a realizar_prediccion después de 5 segundos


# Función para realizar la predicción con el modelo
def realizar_prediccion():
    # Realizar predicción con el modelo
    probabilidades = hacer_prediccion(imagen_cargada)

    # Mostrar la siguiente pantalla con el resultado
    mostrar_resultado(probabilidades)


# Función para hacer la predicción con el modelo
def hacer_prediccion(imagen):
    # Redimensionar la imagen al tamaño esperado por el modelo
    imagen_redimensionada = imagen.resize((200, 200))  # Cambia el tamaño a 200x200
    imagen_array = img_to_array(imagen_redimensionada)  # Convertir a array
    imagen_array = np.expand_dims(imagen_array, axis=0)  # Añadir dimensión de batch

    # Obtener predicción del modelo
    prediccion = modelo.predict(imagen_array)
    print(f"Predicción raw: {prediccion}")  # Para depuración
    return prediccion[0]  # Retornar las probabilidades


# Función para mostrar la imagen cargada y la predicción
def mostrar_resultado(probabilidades):
    global clase_predicha_global
    # Limpiar la ventana
    for widget in ventana.winfo_children():
        widget.destroy()

    # Mostrar la imagen cargada a una dieciseisava parte de su tamaño original
    ancho_original, alto_original = imagen_original.size
    nuevo_tamano = (ancho_original // 16, alto_original // 16)  # Una dieciseisava parte del tamaño original
    imagen_tk = ImageTk.PhotoImage(imagen_original.resize(nuevo_tamano))  # Redimensionar
    label_imagen = Label(ventana, image=imagen_tk, bg='#67b93e')
    label_imagen.image = imagen_tk  # Mantener una referencia de la imagen
    label_imagen.pack(pady=20)

    # Mostrar las probabilidades de cada clase
    texto_probabilidades = f"Probabilidades por clase:\nClase 0: {probabilidades[0]:.2f}\nClase 1: {probabilidades[1]:.2f}\nClase 2: {probabilidades[2]:.2f}\nClase 3: {probabilidades[3]:.2f}"
    label_probabilidades = tk.Label(ventana, text=texto_probabilidades, bg='#67b93e', fg='white', font=("Arial Black", 14))
    label_probabilidades.pack(pady=10)

    # Determinar la clase predicha
    clase_predicha_global = np.argmax(probabilidades)  # Guardar la clase predicha en la variable global
    texto_prediccion = f"Predicción de clase: {clase_predicha_global}"
    label_prediccion = tk.Label(ventana, text=texto_prediccion, bg='#67b93e', fg='white', font=("Arial Black", 14))
    label_prediccion.pack(pady=10)

    # Agregar texto descriptivo de la clase predicha
    descripciones = {
        0: "De siembra a germinación (12 a 15 días)",
        1: "De germinación a floración (27 a 45 días)",
        2: "De la floración a la aparición de la legumbre verde (7 a 15 días)",
        3: "De la floración a la recolección de la semilla (37 a 38 días)"
    }
    
    texto_descripcion = descripciones.get(clase_predicha_global, "Descripción no disponible.")
    label_descripcion = tk.Label(ventana, text=texto_descripcion, bg='#67b93e', fg='white', font=("Arial Black", 14))
    label_descripcion.pack(pady=10)

    # Botón para ir a la pantalla de clasificación de plantas
    boton_siguiente = Button(ventana, text="Siguiente", command=ir_a_clasificacion_plantas, font=("Arial", 12), bg="white", fg="black")
    boton_siguiente.pack(pady=20)


# Función para ir a la quinta pantalla (clasificación de plantas)
def ir_a_clasificacion_plantas():
    # Limpiar la ventana
    for widget in ventana.winfo_children():
        widget.destroy()

    # Configurar la ventana para la clasificación de plantas
    ventana.configure(bg='#67b93e')

    label_texto_clasificacion = tk.Label(ventana, text="Esta sección mejora la imagen y la divide", bg='#67b93e', fg='white', font=("Arial Black", 20))
    label_texto_clasificacion.pack(pady=20)

    boton_identificar = Button(ventana, text="Mejorar Imagen para Modelo", command=identificar_plantas, font=("Arial", 12), bg="white", fg="black")
    boton_identificar.pack(pady=10)


# Función para mejorar la imagen
def enhance_image(image_path):
    # Abrir la imagen usando PIL
    image = Image.open(image_path)

    # Mejora de nitidez
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)  # Ajusta este valor para cambiar la nitidez

    # Mejora de contraste
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)  # Ajusta este valor para cambiar el contraste

    # Mejora de color
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.2)  # Ajusta este valor para cambiar la saturación

    # Convertir imagen a array de OpenCV para la mejora de resolución
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)  # Convertir a BGR para OpenCV

    # Aumentar la resolución usando la interpolación bicúbica
    scale_percent = 150  # Aumentar el tamaño en un 150%
    width = int(image_cv.shape[1] * scale_percent / 100)
    height = int(image_cv.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Redimensionar la imagen
    image_cv = cv2.resize(image_cv, dim, interpolation=cv2.INTER_CUBIC)

    # Convertir de vuelta a RGB para PIL
    image_enhanced = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    # Devolver la imagen mejorada
    return image_enhanced


# Función para dividir la imagen en subimágenes
def split_image(image, size=1280):
    img_width, img_height = image.size

    # Calcular el número de bloques en ambas dimensiones
    num_tiles_x = (img_width + size - 1) // size
    num_tiles_y = (img_height + size - 1) // size

    # Diccionario para almacenar imágenes recortadas y sus posiciones
    image_pieces = {}

    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            start_x = x * size
            start_y = y * size
            end_x = min(start_x + size, img_width)
            end_y = min(start_y + size, img_height)

            # Recortar la imagen
            crop_img = image.crop((start_x, start_y, end_x, end_y))

            # Rellenar si el bloque es más pequeño que el tamaño deseado
            if end_x - start_x < size or end_y - start_y < size:
                padded_img = Image.new('RGB', (size, size), color='black')
                padded_img.paste(crop_img, (0, 0))
                crop_img = padded_img

            # Almacenar la imagen recortada con su posición (fila, columna)
            position = (y, x)  # Fila y columna
            image_pieces[position] = crop_img

    # Imprimir el número total de imágenes recortadas
    print(f"Número total de imágenes recortadas: {len(image_pieces)}")
    return image_pieces, num_tiles_x, num_tiles_y

def reconstruct_image(image_pieces, num_tiles_x, num_tiles_y, size=1280):
    # Crear una imagen vacía del tamaño original reconstruido
    reconstructed_image = Image.new('RGB', (num_tiles_x * size, num_tiles_y * size))

    # Colocar cada subimagen en su posición correcta
    for (y, x), img in image_pieces.items():
        start_x = x * size
        start_y = y * size
        reconstructed_image.paste(img, (start_x, start_y))

    return reconstructed_image

# Variable global para almacenar la subimagen aleatoria
subimagen_aleatoria_global = None
enhanced_image_global = None
# Función para identificar plantas
def identificar_plantas():
    global imagen_cargada, subimagen_aleatoria_global, enhanced_image_global

    # Mejora la imagen cargada
    enhanced_image = enhance_image(ruta_imagen_cargada)
    enhanced_image_global = enhanced_image

    # Dividir la imagen mejorada en subimágenes
    image_pieces, num_tiles_x, num_tiles_y = split_image(enhanced_image, size=1280)

    # Mostrar el número de subimágenes generadas
    texto_subimagenes = f"Número total de subimágenes generadas: {len(image_pieces)}"
    label_subimagenes = tk.Label(ventana, text=texto_subimagenes, bg='#67b93e', fg='white', font=("Arial Black", 14))
    label_subimagenes.pack(pady=10)

    # Mostrar una subimagen generada aleatoria
    subimagen_aleatoria_global = random.choice(list(image_pieces.values()))  # Obtener una subimagen aleatoria
    subimagen_aleatoria_tk = ImageTk.PhotoImage(subimagen_aleatoria_global.resize((1280 // 8, 1280 // 8)))  # Redimensionar para mostrar
    label_subimagen = Label(ventana, image=subimagen_aleatoria_tk, bg='#67b93e')
    label_subimagen.image = subimagen_aleatoria_tk  # Mantener una referencia de la imagen
    label_subimagen.pack(pady=10)

    boton_identificar = Button(ventana, text="Siguiente", command=contar_plantas, font=("Arial", 12), bg="white", fg="black")
    boton_identificar.pack(pady=10)


# En la función contar_plantas
def contar_plantas():
    # Limpiar la ventana
    for widget in ventana.winfo_children():
        widget.destroy()

    # Configurar la ventana para la clasificación de plantas
    ventana.configure(bg='#67b93e')

    label_texto_clasificacion = tk.Label(ventana, text="Esta sección detecta y cuenta las plantas", bg='#67b93e', fg='white', font=("Arial Black", 20))
    label_texto_clasificacion.pack(pady=20)

    boton_identificar = Button(ventana, text="Detectar y contar", command=modelo_roboflow, font=("Arial", 12), bg="white", fg="black")
    boton_identificar.pack(pady=10)

# Función para reconstruir la imagen con las subimágenes procesadas
def reconstruct_image_with_predictions(processed_image_pieces, num_tiles_x, num_tiles_y, size=1280):
    # Crear una imagen vacía del tamaño original reconstruido
    reconstructed_image = Image.new('RGB', (num_tiles_x * size, num_tiles_y * size))

    # Colocar cada subimagen procesada en su posición correcta
    for (y, x), img_np in processed_image_pieces.items():
        # Convertir el array de NumPy a una imagen PIL
        img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

        # Recortar los bordes negros si existen
        non_black_pixels = np.where(np.array(img) != [0, 0, 0])
        if len(non_black_pixels[0]) > 0:  # Si hay píxeles no negros
            min_y = np.min(non_black_pixels[0])
            max_y = np.max(non_black_pixels[0])
            min_x = np.min(non_black_pixels[1])
            max_x = np.max(non_black_pixels[1])

            # Recortar la imagen para eliminar los bordes negros
            img = img.crop((min_x, min_y, max_x + 1, max_y + 1))

        # Pegar la imagen en su lugar original en la imagen reconstruida
        start_x = x * size
        start_y = y * size
        reconstructed_image.paste(img, (start_x, start_y))

    return reconstructed_image


# Función para realizar inferencia con el modelo de Roboflow
def modelo_roboflow():
    global clase_predicha_global, enhanced_image_global
    try:
        if modelos_roboflow.get(clase_predicha_global) is None:
            raise Exception("Se ha detectado una clase invalida")
            
        # Seleccionar la cuenta y el modelo basado en la clase predicha
        cuenta_seleccionada = modelos_roboflow[clase_predicha_global]["cuenta"]
        modelo_seleccionado = modelos_roboflow[clase_predicha_global]["model_id"]
        cliente_seleccionado = clientes_roboflow[cuenta_seleccionada]

        # Dividir la imagen mejorada en subimágenes
        image_pieces, num_tiles_x, num_tiles_y = split_image(enhanced_image_global, size=1280)

        count = 0 #Cantidad de predicciones
        all_sizes = [] # Para almacenar los tamaños de las predicciones
        all_predictions = [] # Para almacenar todas las predicciones
        # Procesar cada subimagen
        processed_pieces = {}  # Diccionario para almacenar subimágenes procesadas
        for i, (position, selected_image) in enumerate(image_pieces.items()):
            image_temp_path = f"temp_image_{i + 1}.jpg"

            # Guardar la subimagen aleatoria temporalmente
            selected_image.save(image_temp_path)  # Guarda temporalmente como un archivo

            # Realizar la inferencia utilizando el cliente de Roboflow con el modelo seleccionado
            result = cliente_seleccionado.infer(image_temp_path, model_id=modelo_seleccionado)
            
            # Procesar resultados
            predicciones = result["predictions"]

            # Asegúrate de que haya predicciones para procesar
            if not predicciones:
                continue
               # raise ValueError("No se detectaron plantas.")
            
            # Filtrar y contar predicciones con certeza del 55% o más
            filtered_predictions = [pred for pred in predicciones if pred['confidence'] >= 0.45]
            all_predictions.extend(filtered_predictions)
            count += len(filtered_predictions)
            
            # Almacenar tamaños de los cuadros
            for prediction in filtered_predictions:
                width = int(prediction['width'])
                all_sizes.append(width)

            # Dibujar bounding boxes en la subimagen
            draw = ImageDraw.Draw(enhanced_image_global)
            for prediccion in filtered_predictions:
                # Obtener las coordenadas del bounding box y la etiqueta
                x = int(prediccion['x'])
                y = int(prediccion['y'])
                width = int(prediccion['width'])
                height = int(prediccion['height'])
                label = prediccion['class']
                confidence = prediccion['confidence']
                
                color = 'green'  # Verde por defecto
                # Clasificar el tamaño del cuadro
                if width < np.mean(all_sizes) * 0.8:
                    color = 'red'  # Rojo
                elif np.mean(all_sizes) * 0.8 <= width <= np.mean(all_sizes) * 1.2:
                    color = 'blue'  # Azul
                
                # Calcular las esquinas del bounding box
                x0 = int(x - width / 2)
                y0 = int(y - height / 2)
                x1 = int(x + width / 2)
                y1 = int(y + height / 2)

                font = ImageFont.load_default()
                font.size = 100

                # Dibujar el rectángulo del bounding box
                draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
                draw.text((x0, y0  - 15), f"{label} ({confidence:.2f})", fill=color, font=font, stroke_width=2,stroke_fill="black")  # Mostrar etiqueta y confianza 

            # Guardar la imagen procesada (con predicciones) en el diccionario
            processed_pieces[position] = cv2.cvtColor(np.array(selected_image), cv2.COLOR_RGB2BGR)

        # Calcular el promedio de tamaño de cuadro al final
        average_size = np.mean(all_sizes)
        print(f"Promedio de tamaño de cuadros: {average_size:.2f}")

        # Mostrar las predicciones en un label
        texto_predicciones = f"Total detectados: {count}\n"
        for prediccion in all_predictions:
            texto_predicciones += f"{prediccion['class']} - Confianza: {prediccion['confidence']:.2f}\n"

          # Reconstruir la imagen final con predicciones
        reconstructed_image = reconstruct_image_with_predictions(processed_pieces, num_tiles_x, num_tiles_y)

        # Redimensionar la imagen para mostrarla en la GUI
        img_tk = ImageTk.PhotoImage(reconstructed_image.resize((1280 // 4, 1280 // 4)))

        # Mostrar la imagen con bounding boxes en la ventana
        panel = tk.Label(ventana, image=img_tk)
        panel.image = img_tk  # Mantener la referencia de la imagen
        panel.pack(pady=20)

        # Mostrar la imagen reconstruida con las predicciones
        dpi = 100  # Resolución en puntos por pulgada
        width, height = reconstructed_image.size  # Tamaño original en píxeles
        scale_factor = 1.05  # Para aumentar el tamaño al 110%
        # Calcular el nuevo tamaño en pulgadas
        new_width = int((width * scale_factor) / dpi) + 20
        new_height = int((height * scale_factor) / dpi) + 20

        resultado = tk.Label(ventana, text=texto_predicciones, justify="left", anchor="w", bg='#67b93e', fg='white', font=("Arial", 12),width=new_width, height=new_height)
        resultado.pack(pady=10)

        enhanced_image_global.show()

    except Exception as e:
        # En caso de error, mostrarlo en la interfaz
        resultado = tk.Label(ventana, text=f"Error en la predicción: {str(e)}", justify="left", anchor="w", fg="red")
        resultado.pack(pady=10)





# =======================================
#           CÓDIGO PRINCIPAL
# =======================================
# Crear ventana principal
ventana = tk.Tk()
ventana.title("Clasificador de Cultivo de Frijol")
ventana.geometry("1000x600")

# Cambiar color de fondo (R:103, G:185, B:62)
ventana.configure(bg='#67b93e')

# Variable global para almacenar la imagen cargada, la original y la predicción
imagen_cargada = None
ruta_imagen_cargada = None
imagen_original = None  # Para guardar la imagen original en memoria
clase_predicha_global = None  # Para guardar el valor de la clase predicha

# Cargar la imagen en la primera pantalla
ruta_imagen = fr"{URL_MODELO}\Imagenes\LOGO_UIS.png"
imagen = Image.open(ruta_imagen)
imagen = imagen.resize((400, 200), Image.Resampling.LANCZOS)
imagen_tk = ImageTk.PhotoImage(imagen)

label_imagen = Label(ventana, image=imagen_tk, bg='#67b93e')
label_imagen.image = imagen_tk
label_imagen.pack(pady=20)

# Iniciar la ventana principal
boton_comenzar = Button(ventana, text="Comenzar", command=ir_a_siguiente, font=("Arial", 12), bg="white", fg="black")
boton_comenzar.pack(pady=10)

# Inicializar los clientes con las respectivas API keys para cada cuenta
clientes_roboflow = {
    "cuenta_1": InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key="zLiTpszg7trBw8EHXpWb"),
    "cuenta_2": InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key="n5NEdn8clXLQedSBVRh1"),
    "cuenta_3": InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key="jXtDT89qtJkli32ITs3o"),
    "cuenta_4": InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key="o8gQm6Mv42JusM2PMTRu")
}

# Diccionario para seleccionar el modelo y la cuenta según la clase predicha
modelos_roboflow = {
    0: {"cuenta": "cuenta_1", "model_id": "model_id_clase_0"},  # Cuenta y modelo para clase 0
    1: {"cuenta": "cuenta_2", "model_id": "beans_detection-sutjq/2"},  # Cuenta y modelo para clase 1
    2: {"cuenta": "cuenta_3", "model_id": "frijol-zsnha/9"},  # Cuenta y modelo para clase 2
    3: {"cuenta": "cuenta_4", "model_id": "beans_detection-bpfrv/3"}   # Cuenta y modelo para clase 3
}

# Iniciar el bucle de la ventana
ventana.mainloop()
