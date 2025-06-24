import dspy
from utils.human_llm import HumanLLM

class HumanLLMAdapter(dspy.dsp.LM):
    def __init__(self, human_llm: HumanLLM):
        super().__init__("human-llm")
        self.human_llm = human_llm

    def __call__(self, prompt: str, **kw):
        res = self.human_llm.invoke(
            user_message=prompt,
            return_message_content_only=True,
            prompt_directory=None,
            **kw,
        )

        queries = res
        # if self.human_llm.agent_name == "question_answering_lm":
        #     queries = queries.split('\n')  # Format as expected by QuestionToQuery

        return queries


    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        return ""

    def __repr__(self)->str:
        return f"HumanLLMAdapter: {self.human_llm.agent_name}>"