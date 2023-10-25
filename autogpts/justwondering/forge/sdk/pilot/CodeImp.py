from typing import Callable, Dict
from forge.sdk.pilot.utils.agent_model import AgentOutput
from forge.sdk.pilot.utils.base_agent import BaseAgent
from pydantic import BaseModel, Field
from typing import Type, Optional
from forge.sdk.pilot.helpers.agents.CodeMonkey import CodeMonkey
from forge.sdk.pilot.helpers.Project import Project
from typing import Type, Optional, List
import os


class CodeMonkeyRefactoredArgs(BaseModel):
    project_name: str = Field(...,
                              description="CodeMokey will use the this to create a new project in the current directory")
    development_plan: List[str] = Field(...,
                                        description="development plan for the CodeMonkey to run")
    mokey: Optional[Type[CodeMonkey]] = None


class CodeMonkeyRefactored(BaseAgent):
    version: str = '0.0.1'
    name: str = 'CodeMonkey'
    type: str = 'react'
    target_tasks: List = ['create a new project and update the project']
    prompt_template: str = ''
    plugins: List = []
    project: Optional[Type[Project]] = None
    description: str = 'CodeMonkey is a developer agent that can implement code changes according to the development plan'
    args_schema: Optional[Type[BaseModel]] = CodeMonkeyRefactoredArgs

    def stream(self, *args, **kwargs) -> AgentOutput:
        return None

    def _compose_plugin_description(self) -> str:
        prompt = f"save_files : save files to the codebase in \n {self.project.get_directory_tree}\n"
        prompt += f"get_files: get file content from the files in codebase. \n"

    def compose_prompt(self, template_name, instruction):
        template_args = dict({
            "step_description": instruction,
            "step_index": 0,
            "directory_tree": self.project.get_directory_tree(True),
        })
        return open(f'pilot/prompt_templates/development/{template_name}.prompt').read().format(**template_args)

    def implement_code_changes(self, instruction):

        self.monkey.implement_code_changes(None, instruction)

        return self.project.get_files()

        # changes = self._extract_changes_from_response(response)

        # for file_data in changes:
        #     self.project.save_file(file_data)

        return response

    def postprocess_response(self, response):
        FUNCTION_CALLS_LIST = self._format_func_call_list()
        if 'function_calls' in response and FUNCTION_CALLS_LIST['function_calls'] is not None:
            response = FUNCTION_CALLS_LIST['function_calls'][response['function_calls']['name']](
                **response['function_calls']['arguments'])
        elif 'text' in response:
            response = response['text']
        return response

    def run(self, project_name, development_plan):
        # tool_description = self._compose_plugin_description()
        # self.description += f"and able to build codebase and update codebase with tools:\n{tool_description}"
        self.project = Project(os.path.join(
            os.environ.get("PROJECT_DIR"), project_name))
        self.monkey = CodeMonkey(self.project, None)
        output = self.implement_code_changes(development_plan)
        return output

    def _extract_changes_from_response(self, response):
        # Parse response to extract file changes
        changes = self.postprocess_response(response)
        return changes

    def _format_function_map(self):
        return {
            'get_files': self.project.get_files,
            'save_file': self.project.save_file
        }
