from src.models.mlp import TinyMLP
from src.models.cnn2d import CNN2DAntiSpoof
from src.models.wav2vec2 import Wav2Vec2AntiSpoof

MODELS = {
    "mlp": TinyMLP,
    "cnn2d": CNN2DAntiSpoof,
    "wav2vec2": Wav2Vec2AntiSpoof,
}

# Hyperparamètres par défaut, utilisés si le modèle n'en spécifie pas d'autres.
DEFAULT_HYPERPARAMS = {
    "batch_size": 32,
    "learning_rate": 1e-3,
}

# Hyperparamètres spécifiques : wav2vec2 part de poids déjà entraînés, donc il faut
# un LR beaucoup plus prudent (sinon on détruit la représentation pré-entraînée en
# quelques batches), et un batch plus petit car le modèle est ~100x plus gros que
# le CNN en nombre de paramètres (94M vs quelques centaines de milliers).
MODEL_HYPERPARAMS = {
    "wav2vec2": {"batch_size": 8, "learning_rate": 2e-5},
}


def build_model(name):
    if name not in MODELS:
        raise ValueError(f"Modèle inconnu : {name}. Choix possibles : {list(MODELS.keys())}")
    return MODELS[name]()


def get_hyperparams(name):
    params = dict(DEFAULT_HYPERPARAMS)
    params.update(MODEL_HYPERPARAMS.get(name, {}))
    return params