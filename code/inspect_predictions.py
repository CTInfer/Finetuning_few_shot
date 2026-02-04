import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


"""
Script pour inspecter visuellement les prédictions du modèle.
Affiche le prompt complet et la réponse brute du modèle.
"""

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = Path("outputs/qwen2_5_7b_nli_lora")
TEST_PATH = Path("NLI4CT/Gold_test_formatted.jsonl")

# Nombre d'exemples à inspecter
NUM_EXAMPLES = 10


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


# ---------------------------------------------------------------------------
# Génération
# ---------------------------------------------------------------------------

def generate_response(model, tokenizer, messages, max_new_tokens=50):
    """Génère une réponse pour un exemple et retourne le prompt et la réponse."""
    # Construire le prompt (sans la réponse de l'assistant)
    prompt_messages = messages[:-1]  # Enlever le dernier message (la réponse)
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Tokenizer
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Génération
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Décoder la réponse générée
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    generated_text = generated_text.strip()
    
    return prompt_text, generated_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("="*80)
    print("INSPECTION DES PRÉDICTIONS DU MODÈLE")
    print("="*80)
    
    # Charger le modèle
    print(f"\nChargement du modèle depuis {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Modèle chargé")
    
    # Charger les données de test
    print(f"\nChargement des données depuis {TEST_PATH}...")
    test_samples = load_jsonl(TEST_PATH)[:NUM_EXAMPLES]
    print(f"Nombre d'exemples à inspecter : {len(test_samples)}")
    
    # Inspecter chaque exemple
    for i, sample in enumerate(test_samples, 1):
        messages = sample["messages"]
        true_label = messages[-1]["content"]
        
        print("\n" + "="*80)
        print(f"EXEMPLE {i}/{len(test_samples)}")
        print("="*80)
        
        # Afficher le vrai label
        print(f"\n📌 VRAI LABEL: {true_label}")
        
        # Générer la réponse
        prompt_text, generated_text = generate_response(model, tokenizer, messages)
        
        # Afficher le prompt
        print("\n" + "-"*80)
        print("PROMPT (ce que le modèle voit) :")
        print("-"*80)
        print(prompt_text)
        
        # Afficher la réponse brute
        print("\n" + "-"*80)
        print("RÉPONSE BRUTE DU MODÈLE :")
        print("-"*80)
        print(repr(generated_text))  # repr() pour voir les caractères spéciaux
        
        # Afficher la réponse formatée
        print("\n" + "-"*80)
        print("RÉPONSE FORMATÉE :")
        print("-"*80)
        print(generated_text)
        
        # Vérifier si c'est correct
        is_correct = (generated_text.strip() == true_label)
        status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        print(f"\n{status}")
        
        # Séparateur visuel
        print("\n")
    
    print("\n" + "="*80)
    print("Inspection terminée")
    print("="*80)


if __name__ == "__main__":
    main()
