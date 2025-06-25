import json
import os, sys
from types import SimpleNamespace
from typing import Union
from langchain_openai import ChatOpenAI
from pathlib import Path
root_path = str(Path(__file__).resolve().parents[2])  # Go up 2 levels to root
sys.path.insert(0, root_path)


from opto.utils.llm import AbstractModel
from utils.human_llm import HumanLLM
from utils.human_llm_config import HumanLLMConfig
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage

class HumanLLM_Trace(AbstractModel):
    def __init__(self, model: Union[str, None] = None, reset_freq: Union[int, None] = None, **kwargs):
        factory = lambda: self._factory(model, **kwargs)
        super().__init__(factory, reset_freq)
        self.model_name = model
        self.is_UI_mode = eval(os.environ.get('UI_MODE', "False"))
        config = HumanLLMConfig()
        config.use_websocket = self.is_UI_mode
        config.common_vectordb_config.embedding_function = "text-embedding-ada-002"
        config.initialize()

    @classmethod
    def _factory(cls, model_name: str, **outer_kwargs):
        # return lambda *args, **kwargs: cls().run(args, **{**cls().kwargs, **kwargs})
        return lambda *args, **call_kwargs: cls().run(model_name, args, **{**outer_kwargs, **call_kwargs})
    
    def run(self, model_name, args, **kwargs):
        res =  HumanLLM(
            agent_name="trace_llm",
            automation=not self.is_UI_mode,
            llmORchains_list=HumanLLMConfig().get_llmORchains_list(),
        ).invoke(
            original_input_messages=[
                SystemMessage(content=kwargs["messages"][0]["content"]),
                HumanMessage(content=kwargs["messages"][1]["content"])
            ],
            prompt_directory=None,
            default_llm_function=ChatOpenAI(
                    model=model_name or "gpt-4o-mini",
                    cache=False,
                    **kwargs
                ),
            *args,
            **kwargs
        )

        if "response_format" in kwargs and kwargs["response_format"]["type"] == "json_object":
            res = ([json.loads(x) for x in res[0]]);
        
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=msg, role="assistant"
                    )
                ) for msg in res
            ]
        )
        return response
    
    @property
    def model(self):
        """
        response = litellm.completion(
            model=self.model,
            messages=[{"content": message, "role": "user"}]
        )
        """
        return lambda *args, **kwargs: self._model(*args, **kwargs)
