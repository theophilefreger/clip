from io import BytesIO
from sys import stderr
import typer
import torch
import torch.nn as nn  # Importation ajoutée ici
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

from clip_server.model.clip_model import CLIPModel
from clip_server.model.tokenization import Tokenizer
from sist2 import Sist2Index, serialize_float_array, print_progress


# Utilisation d'InterpolationMode pour une compatibilité accrue
BICUBIC = InterpolationMode.BICUBIC if hasattr(InterpolationMode, 'BICUBIC') else Image.BICUBIC

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using compute device {DEVICE}")

# Nouvelle fonction de transformation adaptée à CLIP
def get_transform(image_path):
    transform = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return transform(image)

def load_tag_embeddings(tag_file, model, tokenizer):
    with open(tag_file) as f:
        tags = [line.strip() for line in f]

    tokenized = tokenizer(tags, context_length=77, truncate=True)
    text_tokenized = tokenized['input_ids'].to(DEVICE)
    attention_mask = tokenized['attention_mask'].to(DEVICE)

    with torch.no_grad():
        tag_embeddings = model.encode_text(text_tokenized, attention_mask)

    print(f"Pre-computed embeddings for {len(tags)} tags")
    return tag_embeddings, tags

def main(index_file, clip_model: str = "M-CLIP/XLM-Roberta-Large-Vit-B-16Plus", tags_file: str = "general.txt", num_tags: int = 1, color="#dcd7ff"):
    model = CLIPModel(clip_model)
    tokenizer = Tokenizer(clip_model)
    cosine_sim = nn.CosineSimilarity()

    index = Sist2Index(index_file)
    clip_version = index.get("clip_version", default=0)

    index.register_model(
        id=1,
        name="CLIP",
        url="https://raw.githubusercontent.com/simon987/sist2-models/main/clip/models/clip-vit-base-patch32-q8.onnx",
        path="idx_512.clip",
        size=512,
        type="flat"
    )

    where = f"version > {clip_version} AND ((SELECT name FROM mime WHERE id=document.mime) LIKE 'image/%' OR " \
            f"(SELECT name FROM mime WHERE id=document.mime) LIKE 'video/%')"
    total = index.document_count(where)
    done = 0

    for doc in index.document_iter(where):
        try:
            image_transformed = None

            if doc.parent or doc.mime.startswith("video/"):
                tn = index.get_thumbnail(doc.id)
                if not tn:
                    raise Exception("Could not find thumbnail")
                image = Image.open(BytesIO(tn))
                image_transformed = get_transform(BytesIO(tn))
            else:
                image = Image.open(doc.path)
                image_transformed = get_transform(doc.path)

            # Vérification et correction des dimensions de l'image si nécessaire
            if image_transformed is not None:
                if image_transformed.size(0) != 3 or image_transformed.size(1) != 224 or image_transformed.size(2) != 224:
                    raise ValueError("Image dimensions are incorrect")

                image_transformed = image_transformed.unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    embeddings = model.encode_image(image_transformed)

                encoded = serialize_float_array(embeddings.cpu().detach().numpy()[0])

                index.upsert_embedding(doc.id, 0, None, 1, encoded)

                print(f"Generated embeddings for {doc.rel_path}")
                done += 1
                print_progress(done=done, count=total)
            else:
                raise ValueError("Image transformation failed")

        except Exception as e:
            print(f"Could not process image {doc.rel_path}: {e}", file=stderr)
            continue

    index.set("clip_version", index.versions[-1].id)

    print("Syncing tag table")
    index.sync_tag_table()
    index.commit()

    print("Done!")

if __name__ == "__main__":
    typer.run(main)
