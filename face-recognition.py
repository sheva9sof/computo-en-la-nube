import requests
from typing import Any, Dict, List

# Reemplaza con tus valores reales
ENDPOINT = "xxx"
API_KEY = "xxx"

# Imagen pública accesible por internet
IMAGE_URL = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/ComputerVision/Images/faces.jpg"


def detect_faces(endpoint: str, api_key: str, image_url: str) -> Dict[str, Any]:
    """
    Llama a Azure Vision Image Analysis 3.2 para detectar rostros
    en una imagen pública.
    """
    url = endpoint.rstrip("/") + "/vision/v3.2/analyze"
    params = {
        "visualFeatures": "Faces",
    }
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": "application/json",
    }
    body = {
        "url": image_url
    }

    response = requests.post(url, headers=headers, params=params, json=body, timeout=30)
    response.raise_for_status()
    return response.json()


def print_faces_result(result: Dict[str, Any]) -> None:
    faces: List[Dict[str, Any]] = result.get("faces", [])

    print("=" * 70)
    print("RESULTADO DE DETECCIÓN DE ROSTROS")
    print("=" * 70)
    print(f"Total de caras detectadas: {len(faces)}\n")

    if not faces:
        print("No se detectaron rostros en la imagen.")
        return

    for i, face in enumerate(faces, start=1):
        age = face.get("age", "N/D")
        gender = face.get("gender", "N/D")
        rect = face.get("faceRectangle", {})

        print(f"Rostro {i}:")
        print(f"  Edad estimada : {age}")
        print(f"  Género estimado: {gender}")
        print("  Rectángulo facial:")
        print(f"    left   : {rect.get('left', 'N/D')}")
        print(f"    top    : {rect.get('top', 'N/D')}")
        print(f"    width  : {rect.get('width', 'N/D')}")
        print(f"    height : {rect.get('height', 'N/D')}")
        print()


def main() -> None:
    try:
        result = detect_faces(ENDPOINT, API_KEY, IMAGE_URL)
        print_faces_result(result)
    except requests.exceptions.HTTPError as e:
        print("Error HTTP al llamar a Azure:")
        print(e)
        if e.response is not None:
            print("Detalle:")
            print(e.response.text)
    except requests.exceptions.RequestException as e:
        print("Error de conexión o solicitud:")
        print(e)
    except Exception as e:
        print("Error inesperado:")
        print(e)


if __name__ == "__main__":
    main()
