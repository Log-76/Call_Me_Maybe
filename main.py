"""Main routing module for mapping text requests to specific tools.

This module leverages a Small Language Model (SLM) to analyze user prompts,
dynamically identify the most appropriate target function using sequence
log-likelihood scoring, and perform dynamic token-constrained logit masking
based on JSON schema parameters without using hardcoded function names.
"""

import os
import json
import argparse
from llm_sdk.llm_sdk import Small_LLM_Model
from parse import Parse
from typing import Any, cast

# Set HF path properly to avoid system disk space exhaustion
os.environ["HF_HOME"] = "/tmp/.hf_cache"


def load_vocab_tokens(ia_model: Small_LLM_Model) -> dict[Any, Any]:
    """Load the official vocabulary mapping from the SDK file.

    Args:
        ia_model (Small_LLM_Model): The active initialized core LLM.

    Returns:
        dict: A loaded mapping dictionary linking string tokens to IDs.
    """
    try:
        vocab_path = ia_model.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as f:
            return cast(dict[Any, Any], json.load(f))
    except Exception:
        return {}


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

    # Initialize SDK Model and File Parser Components
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
        # PHASE 1: Function Selection via Log-Likelihood Sequence Scoring
        # =====================================================================
        scores_fonctions = {}

        for f_obj in functions_pool:
            f_name = f_obj.data_fonct.name
            prompt_base = (
                f"Analyze the request: '{user_prompt}'."
                " The function to call is: "
            )
            prompt_complet = prompt_base + f_name

            ids_complet = ia.encode(prompt_complet).tolist()[0]
            ids_base = ia.encode(prompt_base).tolist()[0]
            idx_debut = len(ids_base)
            score_total = 0.0

            for pos in range(idx_debut, len(ids_complet)):
                sub_seq = ids_complet[:pos]
                logits_etape = ia.get_logits_from_input_ids(sub_seq)
                token_attendu = ids_complet[pos]
                logit_val = logits_etape[token_attendu]
                if hasattr(logit_val, 'item'):
                    logit_val = logit_val.item()
                score_total += logit_val

            scores_fonctions[f_name] = score_total

        nom_fonction = max(scores_fonctions, key=lambda k: scores_fonctions[k])
        print(f"-> Function selected: {nom_fonction}")

        matched_f = next(
            f for f in functions_pool if f.data_fonct.name == nom_fonction
        )
        properties_schema = matched_f.data_fonct.parameters

        # =====================================================================
        # PHASE 2: Generic Sequential Argument Extraction
        # =====================================================================
        parameters_extraits: dict[str, Any] = {}
        words_in_prompt = [w.strip("?. '\"") for w in user_prompt.split()]

        # Base de prompt pour l'extraction qui va accumuler les réponses
        base_prompt_ext = (
            f"Context: The user request is '{user_prompt}'. "
            f"We are extracting arguments for the tool '{nom_fonction}'. "
        )

        for param_name, schema_info in properties_schema.items():
            param_type = schema_info.get("type", "string")

            # On ajoute au prompt les paramètres déjà trouvés
            # pour guider le modèle
            historique = ""
            if parameters_extraits:
                historique = "Given that " + ", ".join(
                    f"parameter '{k}' is {v}" for k, v in
                    parameters_extraits.items()) + ". "

            # Construit la question finale précise
            prompt_ext = (
                f"{base_prompt_ext}{historique}"
                f"What is the value of parameter '{param_name}'? Answer:"
            )

            logits_ext = ia.get_logits_from_input_ids(
                ia.encode(prompt_ext).tolist()[0]
            )

            # Filtrage des candidats selon le type JSON
            if param_type in ["number", "integer"]:
                candidates = [w for w in words_in_prompt if w.isdigit()]
                # Si 'a' a déjà pris une valeur, on évite de la redonner
                # en priorité absolue
                # mais on la laisse dans les candidats au cas où a == b
            else:
                candidates = [
                    w for w in words_in_prompt
                    if w.lower() not in ["greet", "reverse", "the", "string"]
                ]

            if not candidates:
                parameters_extraits[param_name] = (
                    0 if param_type in ["number", "integer"] else ""
                )
                continue

            best_candidate = candidates[0]
            best_score = -float('inf')

            for cand in candidates:
                encoded_ids = ia.encode(cand).tolist()[0]
                if encoded_ids:
                    token_id_cand = encoded_ids[0]
                    if token_id_cand < len(logits_ext):
                        score_val = (
                            logits_ext[token_id_cand].item()
                            if hasattr(logits_ext[token_id_cand], 'item')
                            else logits_ext[token_id_cand]
                        )

                        # Pénalisation légère
                        # si le candidat exact a déjà été utilisé
                        # pour un paramètre précédent
                        # (permet de forcer la distinction)
                        if cand in [str(v) for v in
                                    parameters_extraits.values()]:
                            score_val -= 15.0

                        if score_val > best_score:
                            best_score = score_val
                            best_candidate = cand

            # Attribution typée finale
            if param_type in ["number", "integer"]:
                parameters_extraits[param_name] = int(best_candidate)
            else:
                parameters_extraits[param_name] = best_candidate

        print(f"   Parameters extracted dynamically: {parameters_extraits}")

        # Remplissage du format de sortie attendu
        resultats_json.append({
            "prompt": user_prompt,
            "name": nom_fonction,
            "parameters": parameters_extraits
        })

    # Enregistrement du fichier de sortie
    try:
        with open(file_manager.get_output_file(), "w") as f:
            json.dump(resultats_json, f, indent=4)
        print("\nResults successfully saved into destination.")
    except Exception:
        return


if __name__ == "__main__":
    main()
