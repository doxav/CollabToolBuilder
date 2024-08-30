from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class TaskIdentified(BaseModel):
    class NextBestTaskModel(BaseModel):
        FunctionName: str = Field(..., description="Name of the next best task identified by the model.")
        Description: str = Field(..., description="Description of the task and its importance.")

    class PerformanceAcceptanceCriteriaModel(BaseModel):
        criteria: Dict[str, Any] = Field(..., description="Criteria for performance acceptance.")

    class DevelopmentPlanModel(BaseModel):
        PlanDepth: int = Field(..., description="Depth of the development plan.")
        Steps: Dict[str, Any] = Field(..., description="Detailed steps for the development plan.")

    class TestModel(BaseModel):
        DocumentID: str = Field(..., description="Unique identifier for the test document.")
        FunctionCall: str = Field(..., description="Function call related to the next best task identified.")

    Reasoning: str = Field(..., description="Analysis of the provided information to determine the next best task.")
    NextBestTask: NextBestTaskModel = Field(..., description="Details of the next best task identified.")
    PerformanceAcceptanceCriteria: PerformanceAcceptanceCriteriaModel = Field(..., description="Criteria for performance acceptance.")
    DevelopmentPlan: DevelopmentPlanModel = Field(..., description="Development plan details.")
    Tests: List[TestModel] = Field(..., description="List of tests associated with the identified tasks.")