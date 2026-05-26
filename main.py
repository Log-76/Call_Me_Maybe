"""Main routing module for mapping text requests to specific tools.

This module leverages a Small Language Model (SLM) to analyze user prompts,
dynamically identify the most appropriate target function using sequence
log-likelihood scoring (Constrained Scoring), and perform token-constrained
logit masking to extract the required arguments before saving the structured
calls into a JSON file.
"""

import os
import json
from llm_sdk.llm_sdk import Small_LLM_Model
from parse import Parse
import torch

# Prevent caching from filling up your home directory
os.environ["HF_HOME"] = "/tmp/.hf_cache"


def main() -> None:
    """Execute the complete Function Calling pipeline on a batch of tests.

    The execution workflow consists of 3 major phases:
    1. System initialization: Loading function schemas and user input cases.
    2. Sequence Iteration: For each individual prompt, evaluates the target
       function and isolates valid tokens to extract parameters.
    3. Exporting: Writing the structured function calls to an output JSON file.
    """
    # 1. Initialize the language model and file parser
    ia = Small_LLM_Model("Qwen/Qwen3-0.6B")
    file = Parse(
        fonct="functions_definition.json",
        input_file="function_calling_tests.json",
        output_file="function_calls.json"
    )

    prompth = file.fonction_input()
    fonct = file.fonction_def()

    # Official list of function names
    noms_fonct = [f.data_fonct.name for f in fonct]
    resultats_json = []

    # 2. Core loop through user prompt sequences
    for i in prompth:
        print(f"\n--- Analyzing request: '{i['prompt']}' ---")

        # =====================================================================
        # PHASE 1: Function Target Selection via Sequence Log-Likelihood
        # =====================================================================
        scores_fonctions = {}

        for f_name in noms_fonct:
            prompt_base = (
                f"Analyze the request: '{i['prompt']}'."
                " The function to call is: "
            )
            prompt_complet = prompt_base + f_name

            # Encode both sequences to isolate the suffix tokens
            ids_complet = ia.encode(prompt_complet).tolist()[0]
            ids_base = ia.encode(prompt_base).tolist()[0]

            # Index boundary where the function name string actually starts
            idx_debut = len(ids_base)
            score_total = 0.0

            for position in range(idx_debut, len(ids_complet)):
                sous_sequence_ids = ids_complet[:position]
                logits_etape = ia.get_logits_from_input_ids(sous_sequence_ids)

                token_attendu = ids_complet[position]
                logit_val = logits_etape[token_attendu]
                if hasattr(logit_val, 'item'):
                    logit_val = logit_val.item()
                score_total += logit_val

            scores_fonctions[f_name] = score_total

        # Select the target function name with maximum cumulative score
        nom_fonction = max(scores_fonctions, key=scores_fonctions.get)
        print(
            "-> Function selected via Constrained Scoring: "
            f"{nom_fonction}"
        )

        # Dictionary to store structured arguments for the final payload
        arguments_extraits = {}

        # =====================================================================
        # PHASE 2: Constrained Token Logit Masking for Argument Extraction
        # =====================================================================

        # --- CASE: fn_add_numbers ---
        if nom_fonction == "fn_add_numbers":
            chiffres_trouves = [
                c.strip('?.') for c in i["prompt"].split()
                if c.strip('?.').isdigit()
            ]

            if len(chiffres_trouves) >= 2:
                # Parameter 'a' Extraction
                prompt_a = (
                    f"In the request '{i['prompt']}',"
                    " what is the first number? Answer:"
                )
                logits_a = ia.get_logits_from_input_ids(
                    ia.encode(prompt_a).tolist()[0]
                )
                scores_a = [-float('inf')] * len(logits_a)
                for num in chiffres_trouves:
                    t_id = ia.encode(num).tolist()[0][0]
                    scores_a[t_id] = (
                        logits_a[t_id].item()
                        if hasattr(logits_a[t_id], 'item')
                        else logits_a[t_id]
                    )
                id_gagnant_a = torch.tensor(scores_a).argmax().item()
                valeur_a = next(
                    int(num) for num in chiffres_trouves
                    if ia.encode(num).tolist()[0][0] == id_gagnant_a
                )

                # Parameter 'b' Extraction
                restants = [
                    num for num in chiffres_trouves
                    if int(num) != valeur_a
                ] or chiffres_trouves
                prompt_b = (
                    f"In the request '{i['prompt']}',"
                    " what is the second number? Answer:"
                )
                logits_b = ia.get_logits_from_input_ids(
                    ia.encode(prompt_b).tolist()[0]
                )
                scores_b = [-float('inf')] * len(logits_b)
                for num in restants:
                    t_id = ia.encode(num).tolist()[0][0]
                    scores_b[t_id] = (
                        logits_b[t_id].item()
                        if hasattr(logits_b[t_id], 'item')
                        else logits_b[t_id]
                    )
                id_gagnant_b = torch.tensor(scores_b).argmax().item()
                valeur_b = next(
                    int(num) for num in restants
                    if ia.encode(num).tolist()[0][0] == id_gagnant_b
                )
            else:
                valeur_a, valeur_b = 0, 0

            arguments_extraits = {"a": valeur_a, "b": valeur_b}
            print(
                f"   Arguments: {arguments_extraits} "
                f"| Execution Result: {valeur_a + valeur_b}"
            )

        # --- CASE: fn_greet ---
        elif nom_fonction == "fn_greet":
            mots_phrase = [m.strip("?. '") for m in i["prompt"].split()]

            prompt_name = (
                f"In the request '{i['prompt']}', "
                "what is the name of the person to greet? Answer:"
            )
            logits_name = ia.get_logits_from_input_ids(
                ia.encode(prompt_name).tolist()[0]
            )
            scores_name = [-float('inf')] * len(logits_name)
            for m in mots_phrase:
                if m.lower() != "greet":
                    t_id = ia.encode(m).tolist()[0][0]
                    scores_name[t_id] = (
                        logits_name[t_id].item()
                        if hasattr(logits_name[t_id], 'item')
                        else logits_name[t_id]
                    )
            id_gagnant_name = torch.tensor(scores_name).argmax().item()

            nom_extrait = next(
                m for m in mots_phrase
                if m.lower() != "greet"
                and ia.encode(m).tolist()[0][0] == id_gagnant_name
            )

            arguments_extraits = {"name": nom_extrait}
            print(
                f"   Arguments: {arguments_extraits} "
                f"| Execution Result: Hello {nom_extrait} !"
            )

        # --- CASE: fn_reverse_string ---
        elif nom_fonction == "fn_reverse_string":
            mots_phrase = [m.strip("?. '") for m in i["prompt"].split()]

            prompt_str = (
                f"In the request '{i['prompt']}', "
                "what is the text string to reverse? Answer:"
            )
            logits_str = ia.get_logits_from_input_ids(
                ia.encode(prompt_str).tolist()[0]
            )
            scores_str = [-float('inf')] * len(logits_str)
            for m in mots_phrase:
                if m.lower() not in ["reverse", "the", "string"]:
                    t_id = ia.encode(m).tolist()[0][0]
                    scores_str[t_id] = (
                        logits_str[t_id].item()
                        if hasattr(logits_str[t_id], 'item')
                        else logits_str[t_id]
                    )

            id_gagnant_str = torch.tensor(scores_str).argmax().item()
            texte_extrait = next(
                m for m in mots_phrase
                if m.lower() not in ["reverse", "the", "string"]
                and ia.encode(m).tolist()[0][0] == id_gagnant_str
            )

            arguments_extraits = {"s": texte_extrait}
            print(
                f"   Arguments: {arguments_extraits} "
                f"| Execution Result: {texte_extrait[::-1]}"
            )

        # Construct payload structure
        donnees_appel = {
            "name": nom_fonction,
            "arguments": arguments_extraits
        }
        resultats_json.append(donnees_appel)

    # 3. Export structured dictionary to disk
    try:
        with open(file.get_output_file(), "w") as f:
            json.dump(resultats_json, f, indent=4)
        print(
            "\nJSON payload structure successfully saved into "
            "function_calls.json!"
        )
    except Exception:
        return None


if __name__ == "__main__":
    main()
