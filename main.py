import os
os.environ["HF_HOME"] = "/tmp/.hf_cache"

from llm_sdk.llm_sdk import Small_LLM_Model
from parse import Parse
import torch


def main() -> None:
    ia = Small_LLM_Model("Qwen/Qwen2.5-0.5B")
    file = Parse(fonct="functions_definition.json",
                 input_file="function_calling_tests.json",
                 output_file="function_calls.json")
    prompth = file.fonction_input()
    fonct = file.fonction_def()
    for i in prompth:
        input_ids = ia.encode(i["prompt"]).tolist()[0]

        logi = ia.get_logits_from_input_ids(input_ids)
        next_token = torch.tensor(logi).argmax().item()
        mot = ia.decode([next_token])
        print(f"{i}{mot}")


main()
