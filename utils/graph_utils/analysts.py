import os
from typing import List, TypedDict

from ..llm_utils import HumanLLMMonitor
from . import helpers_demo
from pydantic import BaseModel, Field
from langgraph.graph import END

analyst = HumanLLMMonitor(agent_name="Create Analysts")

class Analyst(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the analyst.",
    )
    # name: str = Field(
    #     description="Name of the analyst."
    # )
    role: str = Field(
        description="Role/expertise of the analyst in the context of the topic.",
    )
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives.",
    )
    @property
    def persona(self) -> str:
        return f"Role: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )

class GenerateAnalystsState(TypedDict):
    topic: str         # Research topic
    max_analysts: int  # Number of analysts
    human_analyst_feedback: str  # Human feedback
    analysts: List[Analyst]      # List of analysts

def create_analysts(state: GenerateAnalystsState):
    """Create Analysts Agent: Generate a list of analysts in JSON."""
    print("Create_analysts")
    filepath = "create_analysts"
    prompt_file = f"./prompts/{filepath}.txt"
    if not os.path.exists(prompt_file):
        with open(prompt_file, "w") as f:
            f.write(
                "You are tasked with creating a set of AI analyst personas. Your goal is to generate a list of analysts in JSON format and nothing else. "
                "In the user message, you will receive the following values:\n"
                "  - 'TOPIC': the research topic,\n"
                "  - 'FEEDBACK': any editorial feedback,\n"
                "  - 'MAX_ANALYSTS': the maximum number of analysts to generate.\n\n"
                "Review the topic and feedback, identify the top themes, and assign one analyst per theme. "
                "Each analyst must have the following fields: role (string), affiliation (string), and description (string)."
            )
    topic = state['topic']
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get('human_analyst_feedback', '')
    user_message = (
        f"TOPIC: <<< {topic} >>>\n"
        f"MAX_ANALYSTS: <<< {max_analysts} >>>\n"
        f"FEEDBACK: <<< {human_analyst_feedback} >>>\n"
        f"TASK: Generate the set of analysts in JSON format."
    )
    try:
        analysts_response = analyst.CallHumanLLM(
            system_prompt_template=filepath,
            user_message=user_message,
            stream_output=False,
            return_message_content_only=True,
            use_default_llm=False
        )[0]
        # Extract the JSON and convert it to Analyst objects
        analysts_response = helpers_demo.extract_json(helpers_demo.remove_think_tags(analysts_response))
        if isinstance(analysts_response, Perspectives):
            generated_analysts = analysts_response.analysts
        elif isinstance(analysts_response, dict):
            generated_analysts = analysts_response.get('analysts', [])
        elif hasattr(analysts_response, 'analysts'):
            generated_analysts = analysts_response.analysts
        elif isinstance(analysts_response, list):
            generated_analysts = analysts_response
        else:
            generated_analysts = [
                Analyst(
                    role=f"Research Specialist {i+1}",
                    affiliation="Research Institute",
                    description=f"Analyzing aspects of {topic}"
                ) for i in range(max_analysts)
            ]
        # Conversion compatible Pydantic v1/v2
        generated_analysts = [
            Analyst(**(a.model_dump() if hasattr(a, "model_dump") else a.dict() if hasattr(a, "dict") else a))
            for a in generated_analysts
        ]
        return {"analysts": generated_analysts}
    except Exception as e:
        print(f"Error generating analysts: {e}")
        default_analysts = [
            Analyst(
                role=f"Research Specialist {i+1}",
                affiliation="Research Institute",
                description=f"Analyzing aspects of {topic}"
            ) for i in range(max_analysts)
        ]
        return {"analysts": default_analysts}

# def human_feedback(state: GenerateAnalystsState):
#     """No-op node that can be interrupted for human feedback."""
#     print("Human_feedback")
#     pass

def should_continue(state: GenerateAnalystsState):
    print("Should_continue")
    human_analyst_feedback = state.get('human_analyst_feedback', None)
    if human_analyst_feedback:
        return "create_analysts"
    return END
