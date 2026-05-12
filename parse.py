from pydantic import BaseModel, Field
import json


class Parse(BaseModel):
    fonct: str = Field(strict=True)
    input_file: str = Field(strict=True)
    ouput_file: str = Field(strict=True)

    def fonction_def(self):
        try:
            with open(self.fonct, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print("error file not found")
        except json.JSONDecodeError as e:
            print(e.msg)
        except Exception:
            print("error")

    def fonction_input(self):
        try:
            with open(self.input_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print("error file not found")
        except json.JSONDecodeError as e:
            print(e.msg)
        except Exception:
            print("error")

    def get_fonct(self):
        return self.fonct

    def get_input_file(self):
        return self.input_file

    def get_ouput_file(self):
        return self.ouput_file
