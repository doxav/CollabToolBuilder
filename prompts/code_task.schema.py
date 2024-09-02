from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class CodeProposition(BaseModel):
    class HelperFunctionModel(BaseModel):
        function_name: str = Field(..., description="Name of the helper function.")
        code: str = Field(..., description="The full implementation of the helper function (including import statements).")

    class MainFunctionModel(BaseModel):
        function_name: str = Field(..., description="Name of the main function, which should be meaningful and reflect the task.")
        code: str = Field(..., description="The full implementation of the main function (including import statements).")

    Reasoning: str = Field(..., description="Explanation of the implementation assumptions and choices.")
    HelperFunctions: List[HelperFunctionModel] = Field(..., description="List of helper functions implemented.")
    MainFunction: MainFunctionModel = Field(..., description="The main function after the helper functions.")