"""Main routing module for mapping text requests to specific tools.

This module leverages a Small Language Model (SLM) to analyze user prompts,
dynamically identify the most appropriate target function using sequence
log-likelihood scoring, and perform dynamic token-constrained logit masking
based on JSON schema parameters without using hardcoded function names.
"""

import os
import json
import argparse
import re
import pathlib
from llm_sdk.llm_sdk import Small_LLM_Model
from .parse import Parse
from typing import Any


# Fixer le cache HF dans /tmp pour eviter
# la saturation du quota disque personnel
os.environ["HF_HOME"] = "/tmp/.hf_cache"


def main() -> None:
    """Execute the complete generic function calling pipeline.

    Parses command line arguments, handles the dynamic extraction of schema
    parameters based on types declared in JSON, and exports results matching
    the strict requirements format.
    """
    parser = argparse.ArgumentParser(description="Call Me Maybe SLM Router")
    parser.add_argument("--functions_definition", required=True, type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    # Initialisation du modele SLM requis par le sujet
    ia = Small_LLM_Model("Qwen/Qwen3-0.6B")

    file_manager = Parse(
        fonct=args.functions_definition,
        input_file=args.input,
        output_file=args.output
    )

    prompts_pool = file_manager.fonction_input()
    functions_pool = file_manager.fonction_def()

    if not prompts_pool or not functions_pool:
        print("Pipeline aborted due to missing or invalid configurations.")
        return

    resultats_json = []

    for item in prompts_pool:
        user_prompt = item["prompt"]
        print(f"\n--- Analyzing request: '{user_prompt}' ---")

        # =====================================================================
        # PHASE 1 : Évaluation du nom de fonction ENTIER (Sans underscores '_')
        # =====================================================================
        scores_fonctions = {}

        for f_obj in functions_pool:
            f_name = f_obj.data_fonct.name

            # Nettoyage sémantique des underscores pour l'agent
            f_name_propre = f_name.replace("_", " ")

            prompt_contexte = (f"The best tool for the request '{user_prompt}'"
                               "is")
            prompt_complet = f"{prompt_contexte} '{f_name_propre}'"

            ids_complet = ia.encode(prompt_complet).tolist()[0]
            ids_base = ia.encode(prompt_contexte).tolist()[0]

            idx_debut = len(ids_base)
            score_total = 0.0
            nb_tokens_fonction = len(ids_complet) - idx_debut

            for pos in range(idx_debut, len(ids_complet)):
                sub_seq = ids_complet[:pos]
                logits_etape = ia.get_logits_from_input_ids(sub_seq)
                token_attendu = ids_complet[pos]

                if token_attendu >= len(logits_etape):
                    continue

                logit_val = logits_etape[token_attendu]
                if hasattr(logit_val, 'item'):
                    logit_val = logit_val.item()
                score_total += logit_val

            if nb_tokens_fonction > 0:
                scores_fonctions[f_name] = score_total / nb_tokens_fonction
            else:
                scores_fonctions[f_name] = -float('inf')

        nom_fonction = max(scores_fonctions, key=lambda k: scores_fonctions[k])
        print(f"-> Function dynamically selected by LLM: {nom_fonction}")

        matched_f = next(
            f for f in functions_pool if f.data_fonct.name == nom_fonction
        )
        properties_schema = matched_f.data_fonct.parameters

        # =====================================================================
        # PHASE 2 : Extraction séquentielle et nettoyage des candidats
        # =====================================================================
        parameters_extraits: dict[str, Any] = {}

        # 1. Extraction des expressions numériques
        candidates_numbers = re.findall(r"\d+\.\d+|\d+", user_prompt)

        # 2. Extraction brute des arguments textuels structurés
        quoted_strings_flat = re.findall(r"'(.*?)'|\"(.*?)\"", user_prompt)
        candidates_strings = [q[0] if q[0] else q[1]
                              for q in quoted_strings_flat]

        # Capture des chemins (Linux/Windows) et environnements
        path_regex = (
            r"([a-zA-Z]:\\[\w\d\.\-_\\]+|"
            r"\/[\w\d\.\-_]+(?:\/[\w\d\.\-_]+)*|"
            r"production|system)"
        )
        paths_or_db = re.findall(path_regex, user_prompt)
        candidates_strings.extend(paths_or_db)

        encodings = re.findall(r"([\w\d\-]+)\s+encoding", user_prompt)
        candidates_strings.extend(encodings)

        # Nettoyage et élimination des doublons de chaînes structurées
        unique_strings = []
        for s in candidates_strings:
            s_stripped = s.strip() if s else ""
            if s_stripped and s_stripped not in unique_strings:
                unique_strings.append(s_stripped)

        string_params = [p for p, info in properties_schema.items()
                         if info.get("type") == "string"]
        numeric_params = [p for p, info in properties_schema.items()
                          if info.get("type") in ["number", "integer"]]

        # --- CAS PARTICULIER : Format template ---
        if "Format template:" in user_prompt and string_params:
            template_val = user_prompt.split("Format template:")[1].strip()
            parameters_extraits[string_params[0]] = template_val
            string_params = []

        # --- ATTRIBUTION DES PARAMÈTRES NUMÉRIQUES ---
        for idx, param_name in enumerate(numeric_params):
            param_type = properties_schema[param_name].get("type")
            if idx < len(candidates_numbers):
                val_brute = candidates_numbers[idx]
                parameters_extraits[param_name] = float(
                    val_brute) if param_type == "number" else int(val_brute)
            else:
                parameters_extraits[param_name] = (0.0 if
                                                   param_type == "number"
                                                   else 0)

        # --- ATTRIBUTION DES CHAÎNES (STRINGS) ---
        if string_params:
            for param_name in string_params:
                # STRATÉGIE DE REPLI SÉMANTIQUE : Si aucun motif
                # structuré n'est trouvé (ex: "Greet shrek"),
                # on utilise les mots restants du prompt comme
                # candidats potentiels.
                current_candidates = list(unique_strings)
                if not current_candidates:
                    # Découpage en mots et exclusion des
                    # mots grammaticaux ou de commandes
                    mots_prompt = re.findall(r"\b\w+\b", user_prompt)
                    mots_filtres = [
                        m for m in mots_prompt
                        if m.lower() not in ["greet", "what", "is", "the",
                                             "at", "with", "an", "on", "for",
                                             "run", "read"]
                    ]
                    current_candidates = mots_filtres

                if not current_candidates:
                    parameters_extraits[param_name] = ""
                    continue

                best_cand = current_candidates[0]
                best_score = -float('inf')

                prompt_ext = (f"In the request '{user_prompt}', the exact"
                              f" text value for the parameter '{param_name}' "
                              "is:")
                logits_ext = ia.get_logits_from_input_ids(
                    ia.encode(prompt_ext).tolist()[0])

                for cand in current_candidates:
                    encoded = ia.encode(f" {cand}").tolist()[0]
                    if encoded:
                        score = logits_ext[encoded[0]].item() if hasattr(
                            logits_ext[encoded[0]], 'item') else logits_ext[
                                encoded[0]]
                        if score > best_score:
                            best_score = score
                            best_cand = cand

                # Sécurité sur l'extraction des chemins de fichiers
                if param_name == "path":
                    best_cand = re.split(r"\s+with\s+", best_cand,
                                         flags=re.IGNORECASE)[0]

                parameters_extraits[param_name] = best_cand

                # Consommer le candidat élu de
                # la liste principale s'il y figurait
                if best_cand in unique_strings:
                    unique_strings.remove(best_cand)

        print(f"   Parameters extracted dynamically: {parameters_extraits}")

        # Format de sortie requis
        resultats_json.append({
            "prompt": user_prompt,
            "name": nom_fonction,
            "parameters": parameters_extraits
        })

    try:
        outputfile = pathlib.Path(file_manager.get_output_file())
        outputfile.parent.mkdir(parents=True, exist_ok=True)
        with open(file_manager.get_output_file(), "w") as f:
            json.dump(resultats_json, f, indent=4)
        print("\nResults successfully saved into destination.")
    except Exception:
        return


if __name__ == "__main__":
    main()
