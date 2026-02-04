import json
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


"""
Script d'évaluation du modèle finetuné sur le dataset de test.

Utilise le modèle sauvegardé dans OUTPUT_DIR et évalue sur Gold_test_formatted.jsonl.
"""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = Path("outputs/qwen2_5_7b_nli_lora")  # Dossier où le modèle a été sauvegardé
TEST_PATH = Path("NLI4CT/Gold_test_formatted.jsonl")  # Dataset de test
TRAIN_PATH = Path("NLI4CT/train_formatted.jsonl")  # Dataset d'entraînement (pour vérifier)

# Pour un test rapide, on peut limiter le nombre d'exemples
MAX_TEST_SAMPLES = None  # None pour tout utiliser
MAX_TRAIN_SAMPLES_FOR_CHECK = 10  # Nombre d'exemples du train à tester pour vérifier


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
# Prédiction
# ---------------------------------------------------------------------------

def predict(model, tokenizer, messages, max_new_tokens=50, return_raw=False):
    """
    Fait une prédiction sur un exemple au format messages.
    Retourne le label prédit (Entailment ou Contradiction).
    Si return_raw=True, retourne aussi le texte brut généré.
    """
    # Convertir les messages en texte avec le chat template
    # Pour l'inférence, on utilise add_generation_prompt=True pour que le modèle
    # sache qu'il doit répondre en tant qu'assistant
    text = tokenizer.apply_chat_template(
        messages[:-1],  # On enlève le dernier message (la réponse) pour l'inférence
        tokenize=False,
        add_generation_prompt=True,  # Important : dire au modèle qu'il doit répondre
    )
    
    # Tokenizer
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Génération
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Déterministe
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Décoder seulement la partie générée (la réponse de l'assistant)
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    generated_text = generated_text.strip()
    
    # Nettoyer le texte : enlever les retours à la ligne et espaces multiples
    generated_text = " ".join(generated_text.split())
    
    # Le modèle peut générer "assistant Entailment", "Human:", etc.
    # On enlève les préfixes possibles
    if generated_text.lower().startswith("assistant"):
        generated_text = generated_text[len("assistant"):].strip()
    if generated_text.lower().startswith("human:"):
        generated_text = generated_text[len("human:"):].strip()
    # Parfois il peut y avoir "assistant\n" ou "assistant " ou "Human: "
    generated_text = generated_text.replace("assistant\n", "").replace("assistant ", "").strip()
    generated_text = generated_text.replace("Human:\n", "").replace("Human: ", "").replace("Human:", "").strip()
    
    # Extraire le label de manière plus robuste
    # Chercher "Entailment" ou "Contradiction" (insensible à la casse)
    text_lower = generated_text.lower()
    
    # Chercher dans les premiers mots (le label devrait être au début)
    first_words = generated_text.split()[:10]  # Premiers 10 mots
    first_words_lower = " ".join(first_words).lower()
    
    # Vérifier d'abord dans les premiers mots
    if "entailment" in first_words_lower:
        if return_raw:
            return "Entailment", generated_text
        return "Entailment"
    elif "contradiction" in first_words_lower:
        if return_raw:
            return "Contradiction", generated_text
        return "Contradiction"
    
    # Sinon chercher dans tout le texte
    if "entailment" in text_lower:
        if return_raw:
            return "Entailment", generated_text
        return "Entailment"
    elif "contradiction" in text_lower:
        if return_raw:
            return "Contradiction", generated_text
        return "Contradiction"
    
    # Si on ne trouve toujours pas, chercher des variantes
    # Parfois le modèle peut générer "Entail" ou "Contradict"
    if "entail" in text_lower and "contradict" not in text_lower:
        if return_raw:
            return "Entailment", generated_text
        return "Entailment"
    elif "contradict" in text_lower:
        if return_raw:
            return "Contradiction", generated_text
        return "Contradiction"
    
    # Dernier recours : retourner le texte brut pour debug
    # (mais on va quand même essayer de deviner)
    # Si le texte commence par quelque chose qui ressemble à un label...
    if len(generated_text) < 20:  # Texte très court, peut-être juste le label
        if return_raw:
            return generated_text, generated_text
        return generated_text
    
    # Par défaut, on retourne le texte brut (sera compté comme erreur)
    if return_raw:
        return generated_text, generated_text
    return generated_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("="*60)
    print("Évaluation du modèle finetuné")
    print("="*60)
    
    # -----------------------------------------------------------------------
    # Chargement du modèle
    # -----------------------------------------------------------------------
    print(f"\nChargement du modèle depuis {MODEL_DIR}...")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_DIR}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    
    # Déterminer le dtype
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
    
    # -----------------------------------------------------------------------
    # Chargement des données de test
    # -----------------------------------------------------------------------
    print(f"\nChargement des données de test depuis {TEST_PATH}...")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Fichier de test introuvable : {TEST_PATH}")
    
    test_samples = load_jsonl(TEST_PATH)
    if MAX_TEST_SAMPLES is not None:
        test_samples = test_samples[:MAX_TEST_SAMPLES]
    
    print(f"Nombre d'exemples de test : {len(test_samples)}")
    
    # -----------------------------------------------------------------------
    # Prédictions
    # -----------------------------------------------------------------------
    print("\nGénération des prédictions...")
    predictions = []
    true_labels = []
    raw_generations = []  # Pour debug : stocker les générations brutes
    
    for i, sample in enumerate(test_samples):
        if (i + 1) % 50 == 0:
            print(f"  Traité {i + 1}/{len(test_samples)} exemples...")
        
        messages = sample["messages"]
        true_label = messages[-1]["content"]  # Le label est dans le dernier message (assistant)
        true_labels.append(true_label)
        
        # Prédiction (on récupère aussi le texte brut)
        pred_label, raw_text = predict(model, tokenizer, messages, return_raw=True)
        predictions.append(pred_label)
        raw_generations.append(raw_text)
    
    print(f"✓ {len(predictions)} prédictions générées")
    
    # Afficher quelques exemples de génération brute pour debug
    print("\n--- Exemples de générations brutes sur TEST (premiers 5) ---")
    for i in range(min(5, len(raw_generations))):
        print(f"\nExemple {i+1}:")
        print(f"  Vrai label: {true_labels[i]}")
        print(f"  Génération brute: {repr(raw_generations[i][:200])}")  # Premiers 200 caractères
        print(f"  Label extrait: {predictions[i]}")
    
    # Tester aussi sur quelques exemples du train pour voir si le modèle a appris
    if TRAIN_PATH.exists() and MAX_TRAIN_SAMPLES_FOR_CHECK > 0:
        print(f"\n--- Test sur {MAX_TRAIN_SAMPLES_FOR_CHECK} exemples du TRAIN (pour vérifier l'apprentissage) ---")
        train_samples = load_jsonl(TRAIN_PATH)[:MAX_TRAIN_SAMPLES_FOR_CHECK]
        train_correct = 0
        for i, sample in enumerate(train_samples):
            messages = sample["messages"]
            true_label = messages[-1]["content"]
            pred_label, raw_text = predict(model, tokenizer, messages, return_raw=True)
            is_correct = (pred_label == true_label)
            if is_correct:
                train_correct += 1
            print(f"  Exemple train {i+1}: Vrai={true_label}, Prédit={pred_label}, Correct={is_correct}")
        print(f"  Accuracy sur train (échantillon): {train_correct}/{len(train_samples)} = {train_correct/len(train_samples)*100:.1f}%")
    
    # -----------------------------------------------------------------------
    # Calcul des métriques
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("RÉSULTATS")
    print("="*60)
    
    # Accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="weighted", zero_division=0
    )
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    
    # Métriques par classe
    print("\n--- Métriques par classe ---")
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        true_labels, predictions, labels=["Entailment", "Contradiction"], zero_division=0
    )
    
    for label, prec, rec, f1_val, supp in zip(
        ["Entailment", "Contradiction"],
        precision_per_class,
        recall_per_class,
        f1_per_class,
        support
    ):
        print(f"\n{label}:")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1-score: {f1_val:.4f}")
        print(f"  Support: {supp}")
    
    # Matrice de confusion
    print("\n--- Matrice de confusion ---")
    cm = confusion_matrix(true_labels, predictions, labels=["Entailment", "Contradiction"])
    print("                Prédit:")
    print("              Entailment  Contradiction")
    print(f"Entailment      {cm[0][0]:6d}      {cm[0][1]:6d}")
    print(f"Contradiction   {cm[1][0]:6d}      {cm[1][1]:6d}")
    
    # Distribution des prédictions
    print("\n--- Distribution des prédictions ---")
    pred_counter = Counter(predictions)
    true_counter = Counter(true_labels)
    print("Prédictions:", dict(pred_counter))
    print("Vraies labels:", dict(true_counter))
    
    # Exemples d'erreurs (pour analyse qualitative)
    print("\n--- Exemples d'erreurs (premiers 5) ---")
    error_count = 0
    for i, (true, pred) in enumerate(zip(true_labels, predictions)):
        if true != pred and error_count < 5:
            print(f"\nExemple {i+1}:")
            print(f"  Vrai label: {true}")
            print(f"  Prédit: {pred}")
            error_count += 1
    
    print("\n" + "="*60)
    print("Évaluation terminée")
    print("="*60)


if __name__ == "__main__":
    main()
