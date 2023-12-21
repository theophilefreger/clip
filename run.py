from io import BytesIO
from sys import stderr

import typer
import torch
from PIL import Image
from torch import nn
from multilingual_clip import pt_multilingual_clip
import transformers

from sist2 import Sist2Index, serialize_float_array, print_progress

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using compute device {DEVICE}")

# Charger le modèle M-CLIP
MODEL_NAME = "M-CLIP/XLM-Roberta-Large-Vit-L-14"
model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(MODEL_NAME)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

def load_tag_embeddings(tag_file):
    with open(tag_file) as f:
        tags = [line.strip() for line in f]

    text_tokenized = tokenizer(tags, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        tag_embeddings = model.encode_text(text_tokenized)

    print(f"Pre-computed embeddings for {len(tags)} tags")

    return tag_embeddings, tags

def preprocess_image(image):
    transform = pt_multilingual_clip.MultilingualCLIP.get_default_image_transform()
    return transform(image).unsqueeze(0).to(DEVICE)

def main(index_file, clip_model: str = "ViT-B/16", tags_file: str = "general.txt", num_tags: int = 1, color="#dcd7ff"):
    cosine_sim = nn.CosineSimilarity()

    tag_embeddings, tags = load_tag_embeddings(tags_file)

    index = Sist2Index(index_file)

    def process_document(doc):
        try:
            if doc.parent or doc.mime.startswith("video/"):
                tn = index.get_thumbnail(doc.id)
                if not tn:
                    raise Exception("Could not find thumbnail")
                image = Image.open(BytesIO(tn))
            else:
                image = Image.open(doc.path)

            image = preprocess_image(image)
        except Exception as e:
            print(f"Could not load image {doc.rel_path}: {e}", file=stderr)
            return None, None

        with torch.no_grad():
            embeddings = model.encode_image(image)

        return embeddings, doc

    def update_document_tags_and_embeddings(embeddings, doc):
        if embeddings is None:
            return

        if num_tags > 0:
            tags_cos_sim = cosine_sim(tag_embeddings, embeddings).cpu().numpy()
            top_n = tags_cos_sim.argsort()[-num_tags:][::-1]
            top_n_tags = [f"mclip.{tags[i]}.{color}" for i in top_n]

            if "tag" not in doc.json_data:
                doc.json_data["tag"] = top_n_tags
            else:
                doc.json_data["tag"] += [t for t in top_n_tags if t not in doc.json_data["tag"]]

            index.update_document(doc)

        encoded = serialize_float_array(embeddings.cpu().numpy()[0])
        index.upsert_embedding(doc.id, 0, None, 1, encoded)

    # Boucle de traitement des documents
    for doc in index.document_iter(where):
        embeddings, doc = process_document(doc)
        update_document_tags_and_embeddings(embeddings, doc)

        print(f"Processed document: {doc.rel_path}")
        done += 1
        print_progress(done=done, count=total)

    # Mise à jour de l'index et nettoyage
    index.set("mclip_version", model.version)
    index.sync_tag_table()
    index.commit()

    print("Done!")
    
if __name__ == "__main__":
    typer.run(main)
