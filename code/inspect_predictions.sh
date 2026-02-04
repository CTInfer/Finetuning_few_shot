#!/bin/bash
#SBATCH --job-name=evaluate_model
#SBATCH --output=logs/eval_log_%j.out   # Fichier de log (sortie)
#SBATCH --error=logs/eval_log_%j.err    # Fichier d'erreurs
#SBATCH --time=02:00:00                 # Temps max (2h devrait suffire)
#SBATCH --gres=gpu:1                    # 1 GPU pour l'inférence
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G                       # Moins de mémoire nécessaire pour l'inférence

# --- 1. Préparation ---
# On se place dans le bon dossier
cd /mnt/beegfs/home/longuepee/ftctinfer/

# --- 2. Lancement de l'évaluation ---
echo "Démarrage du test.."

# Utiliser directement le Python de l'environnement
/mnt/beegfs/projects/ftctinfer/env_stagiaires/bin/python inspect_predictions.py

echo "Fin de l'évaluation."
