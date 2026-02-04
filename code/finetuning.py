import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

# Mock bitsandbytes si nécessaire pour éviter les erreurs d'import dans PEFT
# On essaie d'abord d'importer bitsandbytes normalement
# IMPORTANT: Si l'import échoue (RuntimeError ou ImportError), on crée un mock IMMÉDIATEMENT
BNB_AVAILABLE = False

# Fonction pour créer le mock de bitsandbytes
def create_bnb_mock():
    """Crée un mock complet de bitsandbytes pour que PEFT puisse fonctionner."""
    class MockLinear8bitLt:
        pass
    
    class MockGlobalOptimManager:
        def __init__(self):
            pass
        def register_parameters(self, *args, **kwargs):
            pass
    
    # Créer un module mock complet
    class MockBNBModule:
        class nn:
            Linear8bitLt = MockLinear8bitLt
            Linear4bit = MockLinear8bitLt  # Pour la 4-bit aussi
        
        class optim:
            GlobalOptimManager = MockGlobalOptimManager
    
    mock_bnb = MockBNBModule()
    
    # Créer un mock pour cextension qui cause souvent des erreurs
    mock_cextension = MagicMock()
    mock_cextension.COMPILED_WITH_CUDA = False
    
    # Injecter le mock dans sys.modules AVANT que PEFT ne l'importe
    sys.modules['bitsandbytes'] = mock_bnb
    sys.modules['bitsandbytes.nn'] = mock_bnb.nn
    sys.modules['bitsandbytes.optim'] = mock_bnb.optim
    sys.modules['bitsandbytes.cextension'] = mock_cextension
    sys.modules['bitsandbytes.cuda_setup'] = MagicMock()
    sys.modules['bitsandbytes.utils'] = MagicMock()
    sys.modules['bitsandbytes.research'] = MagicMock()
    sys.modules['bitsandbytes.research.nn'] = MagicMock()
    return mock_bnb

# Essayer d'importer bitsandbytes
try:
    import bitsandbytes as bnb
    # Vérifier si bitsandbytes fonctionne vraiment en testant l'import des modules critiques
    try:
        # Test pour voir si les modules critiques sont disponibles
        from bitsandbytes.nn import Linear8bitLt
        from bitsandbytes.optim import GlobalOptimManager
        BNB_AVAILABLE = True
    except (RuntimeError, AttributeError, ImportError):
        # bitsandbytes est installé mais ne fonctionne pas (problème CUDA)
        # Créer le mock immédiatement
        BNB_AVAILABLE = False
        create_bnb_mock()
except (ImportError, RuntimeError) as e:
    # bitsandbytes n'est pas installé OU échoue à l'import (problème CUDA)
    # Créer le mock immédiatement pour que PEFT puisse fonctionner
    BNB_AVAILABLE = False
    create_bnb_mock()

# Le mock est déjà créé si nécessaire dans le bloc try/except ci-dessus
# Maintenant on peut importer PEFT en toute sécurité
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


"""
Script de finetuning LoRA avec 4-bit sur Qwen pour la tâche de NLI clinique.

Ce script utilise LoRA (Low-Rank Adaptation) avec quantization 4-bit pour un finetuning efficace.
La 4-bit est OBLIGATOIRE pour pouvoir finetuner de gros modèles.

IMPORTANT: Si bitsandbytes n'est pas disponible ou ne fonctionne pas, le script
échouera avec un message d'erreur explicite. Il faudra alors compiler bitsandbytes
depuis la source (voir compile_bitsandbytes_slurm_fixed.sh).

Hypothèses :
- Les données sont déjà au format JSONL avec un champ "messages" de la forme :
  {
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "Entailment" | "Contradiction"}
    ]
  }
- Fichiers attendus (générés par create_dataset.ipynb) :
  - code/data/train_formatted.jsonl
  - code/data/dev_formatted.jsonl  (optionnel ici, non utilisé pour l'instant)
  - code/data/Gold_test_formatted.jsonl (non utilisé pour l'entraînement)

Ce script est volontairement simple, avec des hyperparamètres codés en dur,
pour tester la configuration du cluster de calcul.
"""


# ---------------------------------------------------------------------------
# Config générale
# ---------------------------------------------------------------------------

# Modèle plus petit pour commencer (0.5B au lieu de 7B)
MODEL_NAME = "/mnt/beegfs/home/longuepee/ftctinfer/model/Qwen2.5-0.5B-Instruct"

DATA_DIR = Path("NLI4CT")
TRAIN_PATH = DATA_DIR / "train_formatted.jsonl"
DEV_PATH = DATA_DIR / "dev_formatted.jsonl"          # non utilisé pour l'instant
TEST_PATH = DATA_DIR / "Gold_test_formatted.jsonl"   # non utilisé pour l'instant

OUTPUT_DIR = Path("outputs/qwen2_5_05b_nli_lora")

# Pour un petit test, on peut limiter le nombre d'exemples
MAX_TRAIN_SAMPLES = None  # None pour tout utiliser (1700 exemples)


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def load_jsonl(path: Path):
    """Charge un fichier JSONL et renvoie une liste de dicts."""
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def build_dataset_from_messages(path: Path, tokenizer, max_samples: int | None = None, max_length: int = 1024):
    """
    Construit un Dataset HF à partir d'un JSONL au format messages.
    On convertit chaque liste de messages en texte avec le template chat du tokenizer.
    """
    raw_samples = load_jsonl(path)

    if max_samples is not None:
        raw_samples = raw_samples[:max_samples]

    texts = []
    for sample in raw_samples:
        messages = sample["messages"]
        # On utilise le chat template de Qwen pour obtenir une seule chaîne de texte
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append({"text": text})

    dataset = Dataset.from_list(texts)
    return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Création du dossier de sortie
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Chargement du tokenizer
    # -----------------------------------------------------------------------
    print(f"Chargement du tokenizer depuis {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Qwen utilise en général un token de fin spécifique
    if tokenizer.pad_token is None:
        # On réutilise le eos_token comme pad_token pour l'entraînement
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------------------------
    # Chargement du modèle avec LoRA + quantization 4-bit (OBLIGATOIRE)
    # -----------------------------------------------------------------------
    use_lora = True
    use_4bit = True  # OBLIGATOIRE pour les gros modèles
    
    # Vérifier que bitsandbytes est disponible
    if not BNB_AVAILABLE:
        print("\n" + "="*80)
        print("❌ ERREUR: bitsandbytes n'est pas disponible ou ne fonctionne pas!")
        print("="*80)
        print("")
        print("La quantization 4-bit est OBLIGATOIRE pour finetuner de gros modèles.")
        print("")
        print("SOLUTION: Compiler bitsandbytes depuis la source")
        print("")
        print("1. Vérifie que GCC >= 7 est disponible:")
        print("   gcc --version")
        print("   module avail gcc  # Si modules disponibles")
        print("")
        print("2. Si GCC >= 7 est disponible, compile bitsandbytes:")
        print("   sbatch compile_bitsandbytes_slurm_fixed.sh")
        print("")
        print("3. Si GCC < 7, contacte l'administrateur du cluster pour:")
        print("   - Installer GCC 9+")
        print("   - Ou fournir un conteneur Docker/Singularity avec GCC")
        print("")
        print("4. Après compilation réussie, relance ce script")
        print("")
        print("="*80)
        raise RuntimeError(
            "bitsandbytes n'est pas disponible. "
            "La quantization 4-bit est obligatoire pour ce script. "
            "Voir le message ci-dessus pour les instructions de compilation."
        )
    
    # Charger le modèle avec 4-bit + LoRA
    print(f"Chargement du modèle {MODEL_NAME} avec LoRA + 4-bit (OBLIGATOIRE)...")
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✓ Modèle chargé avec succès en 4-bit (prêt pour LoRA)")
    except (RuntimeError, ModuleNotFoundError, ImportError) as e:
        error_msg = str(e).lower()
        print("\n" + "="*80)
        print("❌ ERREUR: Échec du chargement du modèle en 4-bit")
        print("="*80)
        print(f"Erreur: {e}")
        print("")
        print("SOLUTION: Compiler bitsandbytes depuis la source")
        print("")
        print("1. Vérifie que GCC >= 7 est disponible:")
        print("   gcc --version")
        print("   module avail gcc  # Si modules disponibles")
        print("")
        print("2. Si GCC >= 7 est disponible, compile bitsandbytes:")
        print("   sbatch compile_bitsandbytes_slurm_fixed.sh")
        print("")
        print("3. Si GCC < 7, contacte l'administrateur du cluster")
        print("")
        print("="*80)
        raise RuntimeError(
            f"Échec du chargement du modèle en 4-bit: {e}. "
            "Voir le message ci-dessus pour les instructions de compilation."
        ) from e

    # -----------------------------------------------------------------------
    # Préparation des données
    # -----------------------------------------------------------------------
    print(f"Chargement des données d'entraînement depuis {TRAIN_PATH}...")
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Fichier d'entraînement introuvable : {TRAIN_PATH}")

    train_dataset = build_dataset_from_messages(
        TRAIN_PATH,
        tokenizer=tokenizer,
        max_samples=MAX_TRAIN_SAMPLES,
        max_length=1024,
    )

    print(f"Nombre d'exemples d'entraînement utilisés : {len(train_dataset)}")

    # -----------------------------------------------------------------------
    # Config LoRA
    # -----------------------------------------------------------------------
    if use_lora:
        print("\nConfiguration LoRA...")
        lora_config = LoraConfig(
            r=16,  # Rank de la décomposition LoRA
            lora_alpha=32,  # Scaling factor
            lora_dropout=0.05,  # Dropout pour éviter l'overfitting
            bias="none",  # Ne pas entraîner les biais
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Modules à adapter pour Qwen
        )
        print("✓ Configuration LoRA prête")
    else:
        lora_config = None

    # -----------------------------------------------------------------------
    # Config entraînement (TrainingArguments pour compatibilité TRL plus ancien)
    # -----------------------------------------------------------------------
    # Avec LoRA, on peut utiliser plus d'epochs et un batch size plus grand
    # car le nombre de paramètres entraînables est beaucoup plus faible.
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=5,  # Plus d'epochs possible avec LoRA (plus léger)
        per_device_train_batch_size=2 if use_lora else 1,  # Batch size plus grand avec LoRA
        gradient_accumulation_steps=4,  # batch effectif de 8 avec LoRA, 4 sans
        learning_rate=2e-4 if use_lora else 5e-5,  # Learning rate un peu plus élevé avec LoRA
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,  # Plus de warmup pour un démarrage plus doux
        logging_steps=50,  # Log tous les 50 steps
        save_strategy="no",  # pas de checkpoints intermédiaires pour ce test
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        gradient_checkpointing=True,
        report_to="none",
    )

    # -----------------------------------------------------------------------
    # Création du trainer TRL (SFTTrainer) avec LoRA + 4-bit
    # -----------------------------------------------------------------------
    print(f"Initialisation du SFTTrainer (LoRA + 4-bit)...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        peft_config=lora_config if use_lora else None,
        args=training_args,
    )

    # -----------------------------------------------------------------------
    # Entraînement
    # -----------------------------------------------------------------------
    print("Démarrage de l'entraînement...")
    trainer.train()
    print("Entraînement terminé.")

    # -----------------------------------------------------------------------
    # Sauvegarde des adapters LoRA
    # -----------------------------------------------------------------------
    print(f"Sauvegarde des adapters LoRA dans {OUTPUT_DIR}...")
    # Pour LoRA, on sauvegarde seulement les adapters (plus léger)
    trainer.model.save_pretrained(OUTPUT_DIR)
    
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Sauvegarde terminée.")


if __name__ == "__main__":
    main()

