from pydantic import BaseModel, Field
from typing import Any
import json


class DataContent(BaseModel):
    prompt: str


class input_test(BaseModel):
    data_input: DataContent


class DataContent_fonct(BaseModel):
    name: str
    description: str
    parameters: Any
    returns: Any = Field(..., alias="return")
    model_config = {"populate_by_name": True}


class fonct_def(BaseModel):
    data_fonct: DataContent_fonct


class Parse(BaseModel):
    fonct: str = Field(strict=True)
    input_file: str = Field(strict=True)
    output_file: str = Field(strict=True)

    def fonction_def(self) -> list | dict | None:
        try:
            with open(self.fonct, "r") as f:
                data = json.load(f)
                c = []
                for i in data:
                    c.append(fonct_def(data_fonct=i))
                return c
        except FileNotFoundError:
            print("error file not found")
        except json.JSONDecodeError as e:
            print(e.msg)
        except Exception as e:
            print("error", e)

    def fonction_input(self) -> list | dict | None:
        try:
            with open(self.input_file, "r") as f:
                data = json.load(f)
                c = []
                for i in data:
                    c.append(i)
            return c
        except FileNotFoundError:
            print("error file not found")
        except json.JSONDecodeError as e:
            print(e.msg)
        except Exception:
            print("error")

    def get_fonct(self) -> str:
        return self.fonct

    def get_input_file(self) -> str:
        return self.input_file

    def get_ouput_file(self) -> str:
        return self.output_file


if __name__ == '__main__':
    c = c = Parse(
        fonct="functions_definition.json",
        input_file="function_calling_tests.json",
        output_file="function_calls.json")
    print(c.fonction_def())
