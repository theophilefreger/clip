from io import BytesIO
import requests
import onnxruntime as ort
from transformers import AutoTokenizer
import typer
from PIL import Image
from torch import nn
import numpy as np
from sist2 import Sist2Index, serialize_float_array, print_progress
import os

# Définition des URLs pour télécharger les modèles ONNX
text_model_url = "https://huggingface.co/immich-app/XLM-Roberta-Large-Vit-B-16Plus/resolve/main/textual/model.onnx?download=true"
vision_model_url = "https://huggingface.co/immich-app/XLM-Roberta-Large-Vit-B-16Plus/resolve/main/visual/model.onnx?download=true"

# Chemins locaux pour enregistrer les modèles ONNX
models_dir = "/models"
text_model_path = os.path.join(models_dir, "text_model.onnx")
vision_model_path = os.path.join(models_dir, "vision_model.onnx")

# Créer le répertoire s'il n'existe pas
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Fonction pour télécharger et sauvegarder les modèles ONNX
def download_model(url, file_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
    else:
        raise ValueError(f"Failed to download the model from {url}")

# Télécharger les modèles ONNX
download_model(text_model_url, text_model_path)
download_model(vision_model_url, vision_model_path)

# Charger les modèles ONNX avec ONNX Runtime
text_model = ort.InferenceSession(text_model_path)
vision_model = ort.InferenceSession(vision_model_path)

# Charger le tokenizer pour le modèle de texte
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

# Fonction pour encoder le texte en utilisant le modèle ONNX
def encode_text(texts):
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="np").input_ids
    return text_model.run(None, {"input_ids": tokens})[0]

# Fonction pour prétraiter et encoder une image en utilisant le modèle ONNX
def preprocess_image(image: Image.Image) -> np.ndarray:
    pass

def encode_image(image_data: np.ndarray):
    return vision_model.run(None, {"image": image_data})[0]

def main(index_file, tags_file: str = "general.txt", num_tags: int = 1, color="#dcd7ff"):
    cosine_sim = nn.CosineSimilarity()
    index = Sist2Index(index_file)

    # Charger et encoder les tags
    with open(tags_file) as f:
        tags = [line.strip() for line in f]
    tag_embeddings = encode_text(tags)

    # Fonction pour traiter un document unique
    def process_document(doc):
        try:
            if doc.parent or doc.mime.startswith("video/"):
                thumbnail = index.get_thumbnail(doc.id)
                if not thumbnail:
                    raise Exception("Thumbnail not found")
                image = Image.open(BytesIO(thumbnail))
            else:
                image = Image.open(doc.path)

            image_data = preprocess_image(image)
            image_embeddings = encode_image(image_data)

        except Exception as e:
            print(f"Error processing document {doc.rel_path}: {e}", file=stderr)
            return None

        return image_embeddings

    # Logique pour traiter chaque document et mettre à jour l'index
    where = f"version > {clip_version} AND ((SELECT name FROM mime WHERE id=document.mime) LIKE 'image/%' OR (SELECT name FROM mime WHERE id=document.mime) LIKE 'video/%')"
    total = index.document_count(where)
    done = 0

    for doc in index.document_iter(where):
        image_embeddings = process_document(doc)

        if image_embeddings is not None:
            # Calculer la similarité cosine entre les tags et les embeddings de l'image
            tags_cos_sim = cosine_sim(np.array(tag_embeddings), np.array(image_embeddings)).cpu().numpy()
            top_n = np.argsort(tags_cos_sim)[-num_tags:]
            top_n_tags = [f"xlmr.{tags[i]}.{color}" for i in top_n]

            # Mettre à jour les tags du document
            doc.json_data["tag"] = top_n_tags
            index.update_document(doc)

            # Mettre à jour les embeddings du document
            encoded = serialize_float_array(image_embeddings)
            index.upsert_embedding(doc.id, 0, None, 1, encoded)

            print(f"Processed document {doc.rel_path}")
            done += 1
            print_progress(done=done, count=total)

    # Mise à jour et synchronisation de l'index
    index.set("xlmr_clip_version", index.versions[-1].id)
    index.sync_tag_table()
    index.commit()

    print("Traitement terminé et index mis à jour.")

# Exécution principale du script
if __name__ == "__main__":
    typer.run(main)
