import io
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any, Optional, Type, Callable
from forge.sdk.workspace import LocalWorkspace

from .prompt_template import PromptTemplate
from .agent_model import AgentType, AgentOutput
from pydantic import BaseModel, create_model

# from gentopia.llm.base_llm import BaseLLM
# from gentopia.memory.api import MemoryWrapper
from rich import print as rprint

from .basetool import BaseTool


class BaseAgent(ABC, BaseModel):
    """Base Agent class defining the essential attributes and methods for an ALM Agent.

    :param name: The name of the agent.
    :type name: str
    :param type: The type of the agent.
    :type type: AgentType
    :param version: The version of the agent.
    :type version: str
    :param description: A brief description of the agent.
    :type description: str
    :param target_tasks: List of target tasks for the agent.
    :type target_tasks: List[str]
    :param prompt_template: PromptTemplate instance or dictionary of PromptTemplate instances. (eg. for ReWOO, two separate PromptTemplates are needed).
    :type prompt_template: Union[PromptTemplate, Dict[str, PromptTemplate]]
    :param plugins: List of plugins available for the agent. PLugins can be tools or other agents.
    :type plugins: List[Any]
    :param args_schema: Schema for arguments, defaults to a model with "instruction" of type str.
    :type args_schema: Optional[Type[BaseModel]]
    :param memory: An instance of MemoryWrapper.
    :type memory: Optional[MemoryWrapper]
    """

    name: str
    type: AgentType
    version: str
    description: str
    prompt_template: Union[PromptTemplate, Dict[str, PromptTemplate]]
    plugins: List[Any]
    args_schema: Optional[Type[BaseModel]] = create_model(
        "ArgsSchema", instruction=(str, ...))
    # workspace: Optional[LocalWorkspace] = None
    # memory: Optional[MemoryWrapper]

    @abstractmethod
    def run(self, *args, **kwargs) -> AgentOutput:
        """Abstract method to be overridden by child classes for running the agent.

        :return: The output of the agent.
        :rtype: AgentOutput
        """
        pass

    @abstractmethod
    def stream(self, *args, **kwargs) -> AgentOutput:
        """Abstract method to be overridden by child classes for running the agent in a stream mode.

        :return: The output of the agent.
        :rtype: AgentOutput
        """
        pass

    def __str__(self):
        """Overrides the string representation of the BaseAgent object.

        :return: The string representation of the agent.
        :rtype: str
        """
        result = io.StringIO()
        rprint(self, file=result)
        return result.getvalue()

    def _compose_plugin_description(self) -> str:
        """
        Compose the worker prompt from the workers.

        Example:
        toolname1[input]: tool1 description
        toolname2[input]: tool2 description
        """
        prompt = ""
        try:
            for plugin in self.plugins:
                prompt += f"{plugin.name}: {plugin.description}\n"
        except Exception:
            raise ValueError("Worker must have a name and description.")
        return prompt

    def _format_func_call_list(self):
        FUNC_CALL_LIST = {
            "definitions": [
            ],
            "function_calls": self._format_function_map()
        }
        for plugin in self.plugins:
            if isinstance(plugin, BaseTool):
                FUNC_CALL_LIST["definitions"].append(format_plugin_schema(plugin))
            else:
                # throw Exception("Not support agent yet")
                raise Exception("Not support agent yet")
                # FUNC_CALL_LIST["definitions"].append(plugin.schema())
                # FUNC_CALL_LIST["function_calls"] += plugin.name + "(" + plugin.name + "_args),"
        return FUNC_CALL_LIST

    def _format_multi_plugins_schema(self):
        """Format list of tools into the open AI function API.

        :param plugins: List of tools to be formatted.
        :type plugins: List[Union[BaseTool, BaseAgent]]
        :return: Formatted tools.
        :rtype: Dict
        """
        schema = {
            'type': 'object',
            'properties': {
                "needs_plan_or_replan": {
                    "type": "boolean",
                    "description": "whether the task needs a plan or replan",
                },
                "tools": {
                    'type': 'array',
                    'description': 'List of actions item that need to be done to complete the task. if a replan is needed, the list will only contain remaining actions which is fail or unknown.',
                    'items': {
                        'type': 'object',
                        'description': 'one of the action items that need to be done to complete the task.',
                        'properties': {
                            'name': {
                                'type': 'string',
                                'enum': [plugin.name for plugin in plugins],
                                'description': 'Name of the tool.',
                            },
                        }
                    },
                }
            }
        }
        for (i, plugin) in enumerate(plugins):
            schema['properties']['tools']['items']['properties'][plugin.name] = plugin.args_schema.schema()
        return {
            'name': 'process_tools',
            'description': 'Process list of tools',
            'parameters': schema,
        }

    def _format_function_schema(self) -> List[Dict]:
        # List the function schema.
        function_schema = []
        for plugin in self.plugins:
            if isinstance(plugin, BaseTool):
                function_schema.append(plugin.schema)
            else:
                function_schema.append(plugin.schema)
        return function_schema

    def _format_function_map(self) -> Dict[str, Callable]:
        """Format the function map for the open AI function API.

        :return: The function map.
        :rtype: Dict[str, Callable]
        """
        # Map the function name to the real function object.
        function_map = {}
        for plugin in self.plugins:
            if isinstance(plugin, BaseTool):
                function_map[plugin.name] = plugin._run
            else:
                function_map[plugin.name] = plugin.run
        return function_map

    def clear(self):
        """
        Clear and reset the agent.
        """
        pass
