from pydantic import BaseModel, Field


class Parse(BaseModel):
    fonct: str = Field(strict=str)
    input_file: str = Field(strict=str)
    ouput_file: str = Field(strict=str)

    def get_fonct(self):
        return self.fonct

    def get_input_file(self):
        return self.input_file

    def get_ouput_file(self):
        return self.ouput_file
