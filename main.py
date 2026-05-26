import os
import json
# Évite de surcharger ton dossier personnel
os.environ["HF_HOME"] = "/tmp/.hf_cache"

from llm_sdk.llm_sdk import Small_LLM_Model
from parse import Parse
import torch


def main() -> None:
    # 1. Initialisation du modèle et du gestionnaire de fichiers
    ia = Small_LLM_Model("Qwen/Qwen3-0.6B")
    file = Parse(fonct="functions_definition.json",
                 input_file="function_calling_tests.json",
                 output_file="function_calls.json")

    prompth = file.fonction_input()
    fonct = file.fonction_def()

    # Liste officielle des noms de fonctions (ex: ['fn_add_numbers',
    # 'fn_greet', 'fn_reverse_string'])
    noms_fonct = [f.data_fonct.name for f in fonct]

    resultats_json = []

    # 2. Boucle principale sur les requêtes utilisateur
    for i in prompth:
        print(f"\n--- Analyse de la requête : '{i['prompt']}' ---")

        # =====================================================================
        # PASSE 1 : Sélection de la fonction par Maximum de Vraisemblance
        # (Séquence)
        # =====================================================================
        scores_fonctions = {}

        # On teste la probabilité de chaque
        # nom de fonction complet dans le contexte
        for f_name in noms_fonct:
            prompt_base = (f"Analyze the request: '{i['prompt']}'."
                           " The function to call is: ")
            prompt_complet = prompt_base + f_name

            # Encodage des deux séquences
            ids_complet = ia.encode(prompt_complet).tolist()[0]
            ids_base = ia.encode(prompt_base).tolist()[0]

            # L'index où commence le nom de la fonction
            idx_debut = len(ids_base)

            # On fait la somme des logits uniquement sur
            # les jetons du nom de la fonction
            score_total = 0.0
            # get_logits_from_input_ids renvoie les logits pour le DERNIER
            # token de la liste fournie.
            # Pour évaluer chaque jeton du nom de la fonction un par un :
            for position in range(idx_debut, len(ids_complet)):
                # On prend les ids jusqu'au jeton précédent
                sous_sequence_ids = ids_complet[:position]
                logits_etape = ia.get_logits_from_input_ids(sous_sequence_ids)

                token_attendu = ids_complet[position]
                logit_val = logits_etape[token_attendu]
                if hasattr(logit_val, 'item'):
                    logit_val = logit_val.item()
                score_total += logit_val

            scores_fonctions[f_name] = score_total

        # L'Argmax choisit la fonction ayant obtenu le plus grand score combiné
        nom_fonction = max(scores_fonctions, key=scores_fonctions.get)
        print("-> Fonction sélectionnée par Constrained Scoring : "
              f"{nom_fonction}")

        # Dictionnaire pour stocker proprement les arguments extraits
        arguments_extraits = {}

        # =====================================================================
        # PASSE 2 : Extraction contrainte des arguments
        # =====================================================================

        # --- CAS : fn_add_numbers ---
        if nom_fonction == "fn_add_numbers":
            chiffres_trouves = [c.strip('?.') for c in i["prompt"].split()
                                if c.strip('?.').isdigit()]

            if len(chiffres_trouves) >= 2:
                # Paramètre 'a'
                prompt_a = (f"In the request '{i['prompt']}',"
                            " what is the first number? Answer:")
                logits_a = ia.get_logits_from_input_ids(
                    ia.encode(prompt_a).tolist()[0])
                scores_a = [-float('inf')] * len(logits_a)
                for num in chiffres_trouves:
                    t_id = ia.encode(num).tolist()[0][0]
                    scores_a[t_id] = logits_a[t_id].item() if hasattr(
                        logits_a[t_id], 'item') else logits_a[t_id]
                id_gagnant_a = torch.tensor(scores_a).argmax().item()
                valeur_a = next(int(num) for num in chiffres_trouves
                                if ia.encode(num).tolist()[0][0] ==
                                id_gagnant_a)

                # Paramètre 'b'
                restants = [num for num in chiffres_trouves
                            if int(num) != valeur_a] or chiffres_trouves
                prompt_b = (f"In the request '{i['prompt']}',"
                            " what is the second number? Answer:")
                logits_b = ia.get_logits_from_input_ids(
                    ia.encode(prompt_b).tolist()[0])
                scores_b = [-float('inf')] * len(logits_b)
                for num in restants:
                    t_id = ia.encode(num).tolist()[0][0]
                    scores_b[t_id] = logits_b[t_id].item() if hasattr(
                        logits_b[t_id], 'item') else logits_b[t_id]
                id_gagnant_b = torch.tensor(scores_b).argmax().item()
                valeur_b = next(int(num) for num in restants
                                if ia.encode(num).tolist()[0][0] ==
                                id_gagnant_b)
            else:
                valeur_a, valeur_b = 0, 0

            arguments_extraits = {"a": valeur_a, "b": valeur_b}
            print(f"   Arguments : {arguments_extraits} "
                  f"| Résultat : {valeur_a + valeur_b}")

        # --- CAS : fn_greet ---
        elif nom_fonction == "fn_greet":
            mots_phrase = [m.strip("?. '") for m in i["prompt"].split()]

            prompt_name = (f"In the request '{i['prompt']}', "
                           "what is the name of the person to greet? Answer:")
            logits_name = ia.get_logits_from_input_ids(
                ia.encode(prompt_name).tolist()[0])
            scores_name = [-float('inf')] * len(logits_name)
            for m in mots_phrase:
                if m.lower() != "greet":
                    t_id = ia.encode(m).tolist()[0][0]
                    scores_name[t_id] = logits_name[t_id].item() if hasattr(
                        logits_name[t_id], 'item') else logits_name[t_id]
            id_gagnant_name = torch.tensor(scores_name).argmax().item()

            nom_extrait = next(m for m in mots_phrase if m.lower() != "greet"
                               and ia.encode(m).tolist()[0][0] ==
                               id_gagnant_name)

            arguments_extraits = {"name": nom_extrait}
            print(f"   Arguments : {arguments_extraits} "
                  f"| Résultat : Bonjour {nom_extrait} !")

        # --- CAS : fn_reverse_string ---
        elif nom_fonction == "fn_reverse_string":
            mots_phrase = [m.strip("?. '") for m in i["prompt"].split()]

            prompt_str = (f"In the request '{i['prompt']}', "
                          "what is the text string to reverse? Answer:")
            logits_str = ia.get_logits_from_input_ids(
                ia.encode(prompt_str).tolist()[0])
            scores_str = [-float('inf')] * len(logits_str)
            for m in mots_phrase:
                if m.lower() not in ["reverse", "the", "string"]:
                    t_id = ia.encode(m).tolist()[0][0]
                    scores_str[t_id] = (logits_str[t_id].item()
                                        if hasattr(logits_str[t_id], 'item')
                                        else logits_str[t_id])

            id_gagnant_str = torch.tensor(scores_str).argmax().item()
            texte_extrait = next(m for m in mots_phrase if m.lower() not in
                                 ["reverse", "the", "string"] and
                                 ia.encode(m).tolist()[0][0] == id_gagnant_str)

            arguments_extraits = {"s": texte_extrait}
            print(f"   Arguments : {arguments_extraits} "
                  f"| Résultat : {texte_extrait[::-1]}")

        # Enregistrement final structuré
        donnees_appel = {
            "name": nom_fonction,
            "arguments": arguments_extraits
        }
        resultats_json.append(donnees_appel)

    # 3. Sauvegarde finale
    try:
        with open("function_calls.json", "w") as f:
            json.dump(resultats_json, f, indent=4)
        print("\nStructure JSON enregistrée avec "
              "succès dans function_calls.json !")
    except Exception:
        return None


main()
