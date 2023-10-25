from dataclasses import dataclass
from typing import Union, NamedTuple


@dataclass
class AgentAction:
    """Agent's action to take.

    :param tool: The tool to invoke.
    :type tool: str
    :param tool_input: The input to the tool.
    :type tool_input: Union[str, dict]
    :param log: The log message.
    :type log: str
    """

    tool: str
    tool_input: Union[str, dict]
    log: str


class AgentFinish(NamedTuple):
    """Agent's return value when finishing execution.

    :param return_values: The return values of the agent.
    :type return_values: dict
    :param log: The log message.
    :type log: str
    """

    return_values: dict
    log: str