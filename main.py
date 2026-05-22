import os
import json
# Évite de surcharger ton dossier personnel
os.environ["HF_HOME"] = "/tmp/.hf_cache"

from llm_sdk.llm_sdk import Small_LLM_Model
from parse import Parse
import torch


def main() -> None:
    # 1. Initialisation du modèle et du gestionnaire de fichiers
    ia = Small_LLM_Model("Qwen/Qwen2.5-0.5B")
    file = Parse(fonct="functions_definition.json",
                 input_file="function_calling_tests.json",
                 output_file="function_calls.json")

    prompth = file.fonction_input()
    fonct = file.fonction_def()

    # Récupération de la liste des noms de fonctions disponibles
    noms_fonct = [f.data_fonct.name for f in fonct]

    resultats_json = []
    # 2. Boucle principale sur tes cas de tests
    for i in prompth:
        print(f"\n--- Analyse de la requête : '{i['prompt']}' ---")
        prompt_lower = i["prompt"].lower()

        # =====================================================================
        # CAS 1 : Déclencheur "sum" -> Fonction fn_add_numbers
        # =====================================================================
        if "sum" in prompt_lower:
            nom_fonction = noms_fonct[0]

            # On isole tous les nombres complets (ex: ["265", "345"])
            chiffres_trouves = [c.strip('?.') for c in i["prompt"].split()
                                if c.strip('?.').isdigit()]

            # --- Extraction du paramètre 'a' (Premier nombre) ---
            prompt_a = (f"In the request '{i['prompt']}',"
                        " what is the first number? Answer:")
            input_ids_a = ia.encode(prompt_a).tolist()[0]
            logits_a = ia.get_logits_from_input_ids(input_ids_a)

            scores_a = [-float('inf')] * len(logits_a)
            for num in chiffres_trouves:
                token_id = ia.encode(num).tolist()[0][0]
                scores_a[token_id] = logits_a[token_id]

            id_gagnant_a = torch.tensor(scores_a).argmax().item()
            valeur_a = next(int(num) for num in chiffres_trouves
                            if ia.encode(num).tolist()[0][0] == id_gagnant_a)

            # --- EXTRACTION DU PARAMÈTRE 'b' (Sécurisée) ---
            # ASTUCE : On crée une liste qui exclut le nombre
            # qu'on vient de trouver pour 'a'
            restants = [num for num in chiffres_trouves
                        if int(num) != valeur_a]

            # Sécurité au cas où l'utilisateur ferait
            # "What is the sum of 5 and 5?"
            # Si les deux nombres d'origine sont identiques,
            # 'restants' serait vide. On le réinitialise.
            if not restants:
                restants = chiffres_trouves

            prompt_b = (f"In the request '{i['prompt']}',"
                        "what is the second number? Answer:")
            input_ids_b = ia.encode(prompt_b).tolist()[0]
            logits_b = ia.get_logits_from_input_ids(input_ids_b)

            scores_b = [-float('inf')] * len(logits_b)
            # On ne masque le tableau QU'AVEC les nombres restants !
            for num in restants:
                token_id = ia.encode(num).tolist()[0][0]
                scores_b[token_id] = logits_b[token_id]

            id_gagnant_b = torch.tensor(scores_b).argmax().item()
            valeur_b = next(int(num) for num in restants
                            if ia.encode(num).tolist()[0][0] == id_gagnant_b)

            # Calcul exact géré par Python
            resultat = valeur_a + valeur_b
            print(f"Fonction : {nom_fonction} |"
                  f" Arguments : a={valeur_a}, b={valeur_b}")
            print(f"Retour de l'outil : {resultat}")

            # Format JSON standard pour le Function Calling
            donnees_appel = {
                "name": nom_fonction,
                "arguments": {"a": valeur_a, "b": valeur_b}
            }
            resultats_json.append(donnees_appel)

        # =====================================================================
        # CAS 2 : Déclencheur "greet" -> Fonction fn_greet
        # =====================================================================
        elif "greet" in prompt_lower:
            nom_fonction = noms_fonct[1]

            # Découpage des mots de la phrase
            mots_phrase = [m.strip("?. '") for m in i["prompt"].split()]

            prompt_name = (f"In the request '{i['prompt']}',"
                           "what is the name of the person? Answer:")
            input_ids_name = ia.encode(prompt_name).tolist()[0]
            logits_name = ia.get_logits_from_input_ids(input_ids_name)

            scores_name = [-float('inf')] * len(logits_name)
            for m in mots_phrase:
                if m.lower() != "greet":
                    token_id = ia.encode(m).tolist()[0][0]
                    scores_name[token_id] = logits_name[token_id]

            id_gagnant_name = torch.tensor(scores_name).argmax().item()
            # Python sélectionne le mot complet intact d'origine
            # ("shrek" ou "john")
            nom_extrait = next(m for m in mots_phrase
                               if m.lower() != "greet" and
                               ia.encode(m).tolist()[0][0] == id_gagnant_name)

            resultat = f"Bonjour {nom_extrait}, ravi de vous rencontrer !"
            print(f"Fonction : {nom_fonction} |"
                  f"Argument : name='{nom_extrait}'")
            print(f"Retour de l'outil : {resultat}")

            donnees_appel = {
                "name": nom_fonction,
                "arguments": {"name": nom_extrait}
            }
            resultats_json.append(donnees_appel)
        # =====================================================================
        # CAS 3 : Déclencheur "reverse" -> Fonction fn_reverse_string
        # =====================================================================
        elif "reverse" in prompt_lower:
            nom_fonction = noms_fonct[2]

            mots_phrase = [m.strip("?. '") for m in i["prompt"].split()]

            prompt_str = (f"In the request '{i['prompt']}', "
                          "what is the text string to reverse? Answer:")
            input_ids_str = ia.encode(prompt_str).tolist()[0]
            logits_str = ia.get_logits_from_input_ids(input_ids_str)

            scores_str = [-float('inf')] * len(logits_str)
            for m in mots_phrase:
                if m.lower() not in ["reverse", "the", "string"]:
                    token_id = ia.encode(m).tolist()[0][0]
                    scores_str[token_id] = logits_str[token_id]

            id_gagnant_str = torch.tensor(scores_str).argmax().item()
            texte_extrait = next(m for m in mots_phrase
                                 if m.lower() not in
                                 ["reverse", "the", "string"] and
                                 ia.encode(m).tolist()[0][0] == id_gagnant_str)

            # Inversion de la chaîne en Python
            resultat = texte_extrait[::-1]
            print(f"Fonction : {nom_fonction} |"
                  f"Argument : s='{texte_extrait}'")
            print(f"Retour de l'outil : {resultat}")

            donnees_appel = {
                "name": nom_fonction,
                "arguments": {"name": nom_extrait}
            }
            resultats_json.append(donnees_appel)
    try:
        with open("function_calls.json", "w")as f:
            json.dump(resultats_json, f, indent=4)
    except Exception:
        return None


main()
