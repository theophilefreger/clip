from io import BytesIO
from sys import stderr
import typer
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

from clip_server.model.clip_model import CLIPModel
from clip_server.model.tokenization import Tokenizer
from sist2 import Sist2Index, serialize_float_array, print_progress

# Configuration du script
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using compute device {DEVICE}")

# Définition des méthodes de transformation d'image
def resize(img: Image.Image, size: int) -> Image.Image:
    if img.width < img.height:
        return img.resize((size, int((img.height / img.width) * size)), resample=Image.BICUBIC)
    else:
        return img.resize((int((img.width / img.height) * size), size), resample=Image.BICUBIC)

def crop(img: Image.Image, size: int) -> Image.Image:
    left = int((img.size[0] / 2) - (size / 2))
    upper = int((img.size[1] / 2) - (size / 2))
    right = left + size
    lower = upper + size
    return img.crop((left, upper, right, lower))

def to_numpy(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB")).astype(np.float32) / 255.0

def normalize(img: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (img - mean) / std

# Fonction de transformation adaptée à CLIP
def get_transform(image_path):
    image = Image.open(image_path)
    if image.mode == 'P':
        image = image.convert('RGBA').convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    resized_image = resize(image, 650)  # Redimensionnement
    cropped_image = crop(resized_image, 640)  # Recadrage
    numpy_image = to_numpy(cropped_image)  # Conversion en tableau numpy
    normalized_image = normalize(numpy_image, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    # Conversion en tenseur PyTorch et réorganisation des dimensions
    return torch.tensor(normalized_image).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

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

    index = Sist2Index(index_file)
    clip_version = index.get("clip_version", default=0)

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
                image_transformed = get_transform(BytesIO(tn))
            else:
                image_transformed = get_transform(doc.path)

            with torch.no_grad():
                embeddings = model.encode_image(image_transformed)

            encoded = serialize_float_array(embeddings.cpu().detach().numpy()[0])
            index.upsert_embedding(doc.id, 0, None, 1, encoded)

            print(f"Generated embeddings for {doc.rel_path}")
            done += 1
            print_progress(done=done, count=total)

        except Exception as e:
            print(f"Could not process image {doc.rel_path}: {e}", file=stderr)
            continue

    index.set("clip_version", index.versions[-1].id)
    index.sync_tag_table()
    index.commit()
    print("Done!")

if __name__ == "__main__":
    typer.run(main)
