import importlib
import importlib.abc
import importlib.machinery
import sys
import time
import inspect
from hllm_utils.human_llm import HumanLLM
from hllm_utils.human_llm_config import HumanLLMConfig
from knowledge_storm.human_llm_adapter import HumanLLMAdapter

_human_llm_parameters = None
_human_llm_config = None
create_human_llm_from_config = None
def _find_caller_identity(skip_modules=None):
    """
    Inspect the call stack and return a tuple (module_name, function_name, filename).
    Skip frames that are inside this patch and inside 'inference' module.
    """
    skip_modules = set(skip_modules or [])
    stack = inspect.stack()
    for frameinfo in stack[1:]:
        frame = frameinfo.frame
        module = inspect.getmodule(frame)
        modname = module.__name__ if module is not None else None
        filename = frameinfo.filename
        func = frameinfo.function
        # Skip frames from inference module and this patch module itself
        if modname in skip_modules or modname in ("inference", __name__):
            continue
        # We found a likely external caller
        return modname, func, filename
    return None, None, None


def patched_curr_cost_est():
    import benchmark.ext.agent_laboratory.inference as agent_lab_inference
    
    original_curr_cost_est = agent_lab_inference.curr_cost_est
    try:
        # Call original function but ensure humanllm costs are handled
        cost = original_curr_cost_est()
        return cost
    except Exception as e:
        print(f"Error in cost estimation: {e}")
        return 0


def patched_query_model(model_str, prompt, system_prompt, openai_api_key=None, anthropic_api_key=None, tries=5, timeout=5.0, temp=None, print_cost=True, version="1.5"):
    import tiktoken
    import benchmark.ext.agent_laboratory.inference as agent_lab_inference
    original_query_model = agent_lab_inference.query_model
    if model_str != "humanllm":
    
        # For non-humanllm models, use original function
        return original_query_model(
            model_str=model_str,
            prompt=prompt,
            system_prompt=system_prompt,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            tries=tries,
            timeout=timeout,
            temp=temp,
            print_cost=print_cost,
            version=version
    )
    # Custom handling for humanllm case
    try:
        # 1) find caller identity
        modname, funcname, filename = _find_caller_identity(skip_modules=("inference", "agents", __name__))


        # Mock implementation for humanllm
        client = None
        for human_llm_config in _human_llm_parameters or []:
            agent_name = human_llm_config.get("agent_name", "")
            if agent_name == funcname:
                client = create_human_llm_from_config(human_llm_config)
                break
        # Fallback: if no name matched, use the first provided dynamic config to
        # ensure trace/dynamic settings are exercised in Agent Laboratory as well.
        if client is None and _human_llm_parameters:
            client = create_human_llm_from_config((_human_llm_parameters or [])[0])
        if client is None:
            client = HumanLLM(
                agent_name=funcname,
                automation=True,
                llmORchains_list=_human_llm_config.get_llmORchains_list()
            )

        human_llm_adapter = HumanLLMAdapter(human_llm=client)
        # Log a minimal trace for visibility
        print(f"[humanllm] AgentLab query -> func={funcname} len(prompt)={len(prompt) if isinstance(prompt, str) else 'NA'}")

        # answers = client.invoke(
        #     system_prompt_template=system_prompt,
        #     user_message=prompt,
        #     prompt_directory=None)
        prompt_parts = [
            f"system: {system_prompt}",
            f"user: {prompt}"
        ]

        response_texts = human_llm_adapter(
            "\n".join(prompt_parts)
        )
        answer = response_texts[0] if response_texts else ""
        
        # Track tokens if needed
        if model_str not in agent_lab_inference.TOKENS_IN:
            agent_lab_inference.TOKENS_IN[model_str] = 0
            agent_lab_inference.TOKENS_OUT[model_str] = 0
            
        encoding = tiktoken.encoding_for_model("gpt-5-nano")
        agent_lab_inference.TOKENS_IN[model_str] += len(encoding.encode(system_prompt + prompt))
        agent_lab_inference.TOKENS_OUT[model_str] += len(encoding.encode(answer))
        
        if print_cost:
            print(f"Current experiment cost = ${patched_curr_cost_est()}, ** Approximate values, may not reflect true cost")
        return answer

    except Exception as e:
        print(f"Error in patched humanllm query: {e}")
        if tries > 1:
            time.sleep(timeout)
            return patched_query_model(
                model_str=model_str,
                prompt=prompt,
                system_prompt=system_prompt,
                openai_api_key=openai_api_key,
                anthropic_api_key=anthropic_api_key,
                tries=tries-1,
                timeout=timeout,
                temp=temp,
                print_cost=print_cost,
                version=version
            )
        raise Exception("Max retries: timeout")


class InferenceLoader(importlib.abc.Loader):
    def __init__(self, orig_loader):
        self.orig_loader = orig_loader

    def create_module(self, spec):
        if hasattr(self.orig_loader, "create_module"):
            return self.orig_loader.create_module(spec)
        return None

    def exec_module(self, module):
        self.orig_loader.exec_module(module)
        # Replace module.query_model right after exec
        try:
            module.query_model = patched_query_model
            # optional log
            print("[inference_import_hook] patched inference.query_model ->")
        except Exception as e:
            print("[inference_import_hook] failed to patch inference:", e)

class InferenceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "inference":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.loader:
            spec.loader = InferenceLoader(spec.loader)
            return spec
        return None
    
# install the finder once
if not any(isinstance(f, InferenceFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, InferenceFinder())
    print("[hook] InferenceFinder installed")
