import io
from typing import Any, Dict, List

import requests
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# =========================
# CONFIGURACIÓN
# =========================
ENDPOINT = "XXX"
API_KEY = "XXX"

# Imagen pública de prueba
IMAGE_URL = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/ComputerVision/Images/faces.jpg"

# Nombre del archivo de salida
OUTPUT_IMAGE = "faces_detected_result.jpg"


def analyze_faces(endpoint: str, api_key: str, image_url: str) -> Dict[str, Any]:
    """
    Envía la imagen pública a Azure Image Analysis 3.2
    para detectar rostros.
    """
    url = endpoint.rstrip("/") + "/vision/v3.2/analyze"
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": "application/json",
    }
    params = {
        "visualFeatures": "Faces"
    }
    body = {
        "url": image_url
    }

    response = requests.post(url, headers=headers, params=params, json=body, timeout=30)
    response.raise_for_status()
    return response.json()


def download_image(image_url: str) -> Image.Image:
    """
    Descarga la imagen desde la URL y la devuelve como objeto PIL.
    """
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def draw_faces(image: Image.Image, faces: List[Dict[str, Any]]) -> Image.Image:
    """
    Dibuja un rectángulo alrededor de cada rostro detectado.
    """
    draw = ImageDraw.Draw(image)

    for index, face in enumerate(faces, start=1):
        rect = face.get("faceRectangle", {})
        left = rect.get("left", 0)
        top = rect.get("top", 0)
        width = rect.get("width", 0)
        height = rect.get("height", 0)

        right = left + width
        bottom = top + height

        # Dibujar rectángulo
        draw.rectangle([(left, top), (right, bottom)], outline="red", width=4)

        # Dibujar etiqueta simple
        label = f"Face {index}"
        text_y = top - 18 if top - 18 > 0 else top + 5
        draw.text((left, text_y), label, fill="red")

    return image


def show_image(image: Image.Image, title: str = "Resultado") -> None:
    """
    Muestra la imagen con matplotlib.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()


def print_faces_info(faces: List[Dict[str, Any]]) -> None:
    """
    Imprime información de los rostros detectados.
    """
    print("=" * 70)
    print("RESULTADO DE DETECCIÓN DE ROSTROS")
    print("=" * 70)
    print(f"Total de caras detectadas: {len(faces)}\n")

    if not faces:
        print("No se detectaron rostros.")
        return

    for i, face in enumerate(faces, start=1):
        rect = face.get("faceRectangle", {})
        age = face.get("age", "N/D")
        gender = face.get("gender", "N/D")

        print(f"Rostro {i}:")
        print(f"  Edad estimada   : {age}")
        print(f"  Género estimado : {gender}")
        print(f"  Left            : {rect.get('left', 'N/D')}")
        print(f"  Top             : {rect.get('top', 'N/D')}")
        print(f"  Width           : {rect.get('width', 'N/D')}")
        print(f"  Height          : {rect.get('height', 'N/D')}")
        print()


def main() -> None:
    if "TU-RECURSO" in ENDPOINT or API_KEY == "TU_API_KEY":
        print("Debes reemplazar ENDPOINT y API_KEY con tus credenciales reales.")
        return

    try:
        # 1. Analizar imagen en Azure
        result = analyze_faces(ENDPOINT, API_KEY, IMAGE_URL)

        # 2. Extraer caras detectadas
        faces = result.get("faces", [])
        print_faces_info(faces)

        # 3. Descargar imagen original
        image = download_image(IMAGE_URL)

        # 4. Dibujar rectángulos de rostros
        image_with_faces = draw_faces(image, faces)

        # 5. Guardar resultado
        image_with_faces.save(OUTPUT_IMAGE)
        print(f"Imagen guardada como: {OUTPUT_IMAGE}")

        # 6. Mostrar resultado
        show_image(image_with_faces, title=f"Caras detectadas: {len(faces)}")

    except requests.exceptions.HTTPError as e:
        print("Error HTTP al llamar a Azure:")
        print(e)
        if e.response is not None:
            print(e.response.text)
    except requests.exceptions.RequestException as e:
        print("Error de red o solicitud:")
        print(e)
    except Exception as e:
        print("Error inesperado:")
        print(e)


if __name__ == "__main__":
    main()
