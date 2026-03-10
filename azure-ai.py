from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.exceptions import HttpResponseError


# =========================
# CONFIGURACIÓN
# =========================
ENDPOINT = "XXX"
API_KEY = "XXX"


def create_client(endpoint: str, api_key: str) -> TextAnalyticsClient:
    """
    Crea y devuelve un cliente de Azure AI Language.
    """
    credential = AzureKeyCredential(api_key)
    return TextAnalyticsClient(endpoint=endpoint, credential=credential)


def get_documents() -> list[dict]:
    """
    Devuelve una lista de documentos más completa para probar
    análisis de sentimiento y extracción de frases clave.
    """
    return [
        {
            "id": "1",
            "language": "es",
            "text": (
                "El servicio al cliente fue excelente. "
                "Me atendieron rápidamente, resolvieron mi problema y el personal fue muy amable."
            ),
        },
        {
            "id": "2",
            "language": "es",
            "text": (
                "La entrega fue un desastre. "
                "El pedido llegó tarde, la caja estaba dañada y además faltaban productos."
            ),
        },
        {
            "id": "3",
            "language": "en",
            "text": (
                "The hotel location was convenient and the room was clean, "
                "but the check-in process was slow and the staff seemed unprepared."
            ),
        },
        {
            "id": "4",
            "language": "fr",
            "text": (
                "Le produit est de bonne qualité et fonctionne correctement, "
                "mais la documentation n'est pas très claire."
            ),
        },
        {
            "id": "5",
            "language": "pt",
            "text": (
                "A aplicação tem uma interface intuitiva, porém apresentou travamentos "
                "quando tentei carregar arquivos grandes."
            ),
        },
        {
            "id": "6",
            "language": "de",
            "text": (
                "Das Restaurant war sauber und das Essen war lecker, "
                "aber der Service war langsam und die Rechnung war zu hoch."
            ),
        },
    ]


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def analyze_sentiment(client: TextAnalyticsClient, documents: list[dict]) -> None:
    """
    Ejecuta análisis de sentimiento y muestra resultados detallados.
    """
    print_header("ANÁLISIS DE SENTIMIENTO")

    try:
        results = client.analyze_sentiment(documents=documents, show_opinion_mining=False)

        for doc, result in zip(documents, results):
            print(f"\nDocumento ID: {doc['id']}")
            print(f"Idioma: {doc['language']}")
            print(f"Texto: {doc['text']}")

            if result.is_error:
                print(f"Error: {result.error.code} - {result.error.message}")
                continue

            print(f"Sentimiento general: {result.sentiment}")
            print("Puntajes de confianza:")
            print(f"  Positivo: {result.confidence_scores.positive:.4f}")
            print(f"  Neutral : {result.confidence_scores.neutral:.4f}")
            print(f"  Negativo: {result.confidence_scores.negative:.4f}")

            print("\nSentimiento por oración:")
            for idx, sentence in enumerate(result.sentences, start=1):
                print(f"  Oración {idx}: {sentence.text}")
                print(f"    Sentimiento: {sentence.sentiment}")
                print(
                    "    Confianza -> "
                    f"Positivo: {sentence.confidence_scores.positive:.4f}, "
                    f"Neutral: {sentence.confidence_scores.neutral:.4f}, "
                    f"Negativo: {sentence.confidence_scores.negative:.4f}"
                )

    except HttpResponseError as e:
        print("Error al ejecutar analyze_sentiment:")
        print(e)


def extract_key_phrases(client: TextAnalyticsClient, documents: list[dict]) -> None:
    """
    Extrae frases clave de los documentos.
    """
    print_header("EXTRACCIÓN DE FRASES CLAVE")

    try:
        results = client.extract_key_phrases(documents=documents)

        for doc, result in zip(documents, results):
            print(f"\nDocumento ID: {doc['id']}")
            print(f"Idioma: {doc['language']}")
            print(f"Texto: {doc['text']}")

            if result.is_error:
                print(f"Error: {result.error.code} - {result.error.message}")
                continue

            print("Frases clave detectadas:")
            if result.key_phrases:
                for phrase in result.key_phrases:
                    print(f"  - {phrase}")
            else:
                print("  No se detectaron frases clave.")

    except HttpResponseError as e:
        print("Error al ejecutar extract_key_phrases:")
        print(e)


def main() -> None:
    """
    Función principal del programa.
    """
    print_header("AZURE AI LANGUAGE - PRUEBA DESDE PYTHON")

    if ENDPOINT == "" or API_KEY == "":
        print("Debes reemplazar TU_ENDPOINT y TU_API_KEY con tus credenciales reales.")
        return

    client = create_client(ENDPOINT, API_KEY)
    documents = get_documents()

    analyze_sentiment(client, documents)
    extract_key_phrases(client, documents)

    print_header("PROCESO FINALIZADO")


if __name__ == "__main__":
    main()
