import dspy
from utils.human_llm import HumanLLM
from utils.human_llm_config import HumanLLMConfig


class HumanLLMAdapter(dspy.dsp.LM):
    def __init__(self, human_llm: HumanLLM = None,*args, **kwargs):
        super().__init__("human-llm")
        if human_llm is None:
            config = HumanLLMConfig()
            config.use_websocket = False
            config.common_vectordb_config.embedding_function = "text-embedding-ada-002"
            config.initialize()
            human_llm = HumanLLM(
                agent_name="Default",
                automation=True,
                llmORchains_list=config.get_llmORchains_list()
            )
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

    def set_agent_name(self, agent_name: str):
        self.human_llm.agent_name = agent_name

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        return ""

    def __repr__(self)->str:
        return f"HumanLLMAdapter: {self.human_llm.agent_name}>"
    
    
