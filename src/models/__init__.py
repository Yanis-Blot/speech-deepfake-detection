from src.models.mlp import TinyMLP
from src.models.cnn2d import CNN2DAntiSpoof

MODELS = {
    "mlp": TinyMLP,
    "cnn2d": CNN2DAntiSpoof,
}


def build_model(name):
    if name not in MODELS:
        raise ValueError(f"Modèle inconnu : {name}. Choix possibles : {list(MODELS.keys())}")
    return MODELS[name]()