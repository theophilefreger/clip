from io import BytesIO
import requests
import onnxruntime as ort
from transformers import AutoTokenizer
import typer
from PIL import Image
import numpy as np
from sist2 import Sist2Index, serialize_float_array, print_progress
import os
from numpy import dot
from numpy.linalg import norm

# Définitions des chemins et URLs des modèles
models_dir = "/models"
text_model_path = os.path.join(models_dir, "text_model.onnx")
vision_model_path = os.path.join(models_dir, "vision_model.onnx")
text_model_url = "https://huggingface.co/immich-app/XLM-Roberta-Large-Vit-B-16Plus/resolve/main/textual/model.onnx?download=true"
vision_model_url = "https://huggingface.co/immich-app/XLM-Roberta-Large-Vit-B-16Plus/resolve/main/visual/model.onnx?download=true"

# Fonctions de téléchargement et de prétraitement
def download_model(url, file_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
    else:
        raise ValueError(f"Failed to download the model from {url}")

def preprocess_image(image: Image.Image, target_size: tuple = (224, 224)) -> np.ndarray:
    image = image.resize(target_size)
    image_array = np.array(image).astype(np.float32) / 255.0
    return image_array

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# Téléchargement des modèles ONNX
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
download_model(text_model_url, text_model_path)
download_model(vision_model_url, vision_model_path)

# Chargement des modèles ONNX et du tokenizer
text_model = ort.InferenceSession(text_model_path)
vision_model = ort.InferenceSession(vision_model_path)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

# Fonctions d'encodage
def encode_text(texts):
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="np").input_ids
    return text_model.run(None, {"input_ids": tokens})[0]

def encode_image(image_data: np.ndarray):
    return vision_model.run(None, {"image": image_data})[0]

# Fonction principale
def main(index_file, tags_file: str = "general.txt", num_tags: int = 1, color="#dcd7ff"):
    index = Sist2Index(index_file)

    # Charger et encoder les tags
    with open(tags_file) as f:
        tags = [line.strip() for line in f]
    tag_embeddings = encode_text(tags)

    # Logique de traitement des documents
    where = "une_condition_adéquate_pour_sélectionner_les_documents"  # Mettez ici votre condition
    total = index.document_count(where)
    done = 0

    for doc in index.document_iter(where):
        try:
            # Chargement de l'image
            if doc.parent or doc.mime.startswith("video/"):
                thumbnail = index.get_thumbnail(doc.id)
                if not thumbnail:
                    raise Exception("Thumbnail not found")
                image = Image.open(BytesIO(thumbnail))
            else:
                image = Image.open(doc.path)

            # Encodage de l'image
            image_data = preprocess_image(image)
            image_embeddings = encode_image(image_data)

            # Calcul de la similarité
            tags_cos_sim = [cosine_similarity(embedding, image_embeddings) for embedding in tag_embeddings]
            top_n = np.argsort(tags_cos_sim)[-num_tags:]
            top_n_tags = [f"xlmr.{tags[i]}.{color}" for i in top_n]

            # Mise à jour des tags du document
            doc.json_data["tag"] = top_n_tags
            index.update_document(doc)

            # Mise à jour des embeddings
            encoded = serialize_float_array(image_embeddings)
            index.upsert_embedding(doc.id, 0, None, 1, encoded)

            print(f"Processed document {doc.rel_path}")
            done += 1
            print_progress(done=done, count=total)

        except Exception as e:
            print(f"Error processing document {doc.rel_path}: {e}")

    # Mise à jour et synchronisation de l'index
    index.set("xlmr_clip_version", index.versions[-1].id)
    index.sync_tag_table()
    index.commit()

    print("Traitement terminé et index mis à jour.")

# Exécution du script
if __name__ == "__main__":
    typer.run(main)
