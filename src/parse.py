"""Data Parsing and Serialization Validation Module using Pydantic.

This module contains the structural schemas and utilities required to parse,
validate, and handle system files containing tool schemas and batch arrays
of execution context queries.
"""

import json
from typing import Any, List, Optional, Dict
from pydantic import BaseModel, Field


class DataContent(BaseModel):
    """Data object mapping raw inner user input text data.

    Attributes:
        prompt (str): The raw string request given by the user.
    """

    prompt: str


class InputTest(BaseModel):
    """Pydantic model wrapping an individual input test context block.

    Attributes:
        data_input (DataContent): Container containing the raw text prompt.
    """

    data_input: DataContent


class DataContentFonct(BaseModel):
    """Schema representing complete structural tool attributes for validation.

    Attributes:
        name (str): The program name identifier of the function.
        description (str): Explanatory documentation of the capabilities.
        parameters (Any): JSON Schema dictionary specifying required fields.
        returns (Any): JSON Schema description of the function's return types.
    """

    name: str
    description: str
    parameters: Any
    returns: Any
    model_config = {"populate_by_name": True}


class FonctDef(BaseModel):
    """Wrapper component encapsulating a single function specification object.

    Attributes:
        data_fonct (DataContentFonct): Validated inner tool attributes.
    """

    data_fonct: DataContentFonct


class Parse(BaseModel):
    """Data Pipeline IO controller for managing serialization files.

    Handles error-resilient deserialization of schemas and data logs while
    isolating underlying system storage configurations.

    Attributes:
        fonct (str): System file path targeting the function definitions.
        input_file (str): System file path targeting the collection of tests.
        output_file (str): Target path for exporting structured JSON runs.
    """

    fonct: str = Field(strict=True)
    input_file: str = Field(strict=True)
    output_file: str = Field(strict=True)

    def fonction_def(self) -> Optional[List[FonctDef]]:
        """Parse and instantiate tool metadata models from a JSON file.

        Returns:
            Optional[List[FonctDef]]: A collection of validated FonctDef
                objects, or None if file processing is interrupted.
        """
        try:
            with open(self.fonct, "r") as f:
                data = json.load(f)
                c = []
                for i in data:
                    c.append(FonctDef(data_fonct=i))
                return c
        except FileNotFoundError:
            print("Error: Functions definition file not found.")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e.msg}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def fonction_input(self) -> Optional[List[Dict[str, Any]]]:
        """Deserialize raw batch execution strings from the input file.

        Returns:
            Optional[List[Dict[str, Any]]]: An array of dictionaries matching
                input payloads on success, or None if access fails.
        """
        try:
            with open(self.input_file, "r") as f:
                data = json.load(f)
                c = []
                for i in data:
                    c.append(i)
            return c
        except FileNotFoundError:
            print("Error: Input file not found.")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e.msg}")
            return None
        except Exception:
            print("Unexpected error during input parsing.")
            return None

    def get_fonct(self) -> str:
        """Retrieve the file system path of the tool schema database.

        Returns:
            str: Configuration path of the source function schemas file.
        """
        return self.fonct

    def get_input_file(self) -> str:
        """Retrieve the file system path containing prompt test strings.

        Returns:
            str: Configuration path of the system input test data file.
        """
        return self.input_file

    def get_output_file(self) -> str:
        """Retrieve the file system destination path configured for export.

        Returns:
            str: Target destination path of the output tracking record file.
        """
        return self.output_file
