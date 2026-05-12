from pydantic import BaseModel, Field


class Parse(BaseModel):
    fonct: str = Field(strict=str)
    input_file: str = Field(strict=str)
    ouput_file: str = Field(strict=str)
