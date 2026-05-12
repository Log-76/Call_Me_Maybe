from pydantic import BaseModel, Field


class Parse(BaseModel):
    def __init__(self, name, bases, dict, /, **kwds):
        super().__init__(name, bases, dict, **kwds)
        fonct : str = Field(strict=str)
        input_file : str = Field(strict=str)
        ouput_file : str = Field(strict=str)

