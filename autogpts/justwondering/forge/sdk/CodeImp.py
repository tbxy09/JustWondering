from typing import Callable, Dict
from forge.sdk.pilot.utils.agent_model import AgentOutput
from forge.sdk.pilot.utils.base_agent import BaseAgent
from forge.sdk.pilot.utils.files import create_directory
from pydantic import BaseModel, Field
from typing import Type, Optional
from forge.sdk.pilot.helpers.agents.CodeMonkey import CodeMonkey
from forge.sdk.pilot.helpers.agents.Developer import Developer
from forge.sdk.pilot.helpers.Project import Project
from typing import Type, Optional, List
import os


class CodeMonkeyRefactoredArgs(BaseModel):
    project_name: str = Field(...,
                              description="CodeMokey will use the this to create a new project")
    development_plan: List[str] = Field(...,
                                        description="development plan for the CodeMonkey to run")


class CodeMonkeyRefactored(BaseAgent):
    version: str = '0.0.1'
    name: str = 'CodeMonkey'
    type: str = 'react'
    target_tasks: List = ['create a new project and update the project']
    prompt_template: str = ''
    plugins: List = []
    monkey: Optional[Type[CodeMonkey]] = None
    project: Optional[Type[Project]] = None
    description: str = 'As an AI tool, you have full-stack working experience. You write code and practice Test-Driven Development (TDD) whenever it is suitable. Your main responsibility is to implement assigned tasks.'
    args_schema: Optional[Type[BaseModel]] = CodeMonkeyRefactoredArgs
    developer: Optional[Type[Developer]] = None

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

    def implement_code_changes(self, instruction, specs=None, convo=None):
        # if self.project is None:
        #     self.project = Project(os.path.join(
        #         os.environ.get("PROJECT_DIR"), "test_project"))
        return self.monkey.implement_code_changes(convo, instruction, specs=specs)

    def postprocess_response(self, response):
        FUNCTION_CALLS_LIST = self._format_func_call_list()
        if 'function_calls' in response and FUNCTION_CALLS_LIST['function_calls'] is not None:
            response = FUNCTION_CALLS_LIST['function_calls'][response['function_calls']['name']](
                **response['function_calls']['arguments'])
        elif 'text' in response:
            response = response['text']
        return response

    def run(self, project_name, specs, development_plan, task_id):
        # tool_description = self._compose_plugin_description()
        # self.description += f"and able to build codebase and update codebase with tools:\n{tool_description}"
        if project_name is None:
            project_name = "test_project"
        if project_name.split(".")[-1] != "":
            project_name = project_name.split(".")[0]
        self.project = Project({})
        # if os.environ.get("PROJECT_DIR") != os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../../workspace")):
        #     raise Exception(
        #         "Please set PROJECT_DIR environment variable to the project directory")
        project_path = create_directory(
            os.environ.get("AGENT_WORKSPACE"), project_name)
        project_path = create_directory(
            project_path, "artifact_out")

        # create_directory(project_path, 'tests')
        self.project.task_id = task_id
        self.project.current_step = 0
        self.project.root_path = project_path
        self.project.get_files_list(task_id)
        self.monkey = CodeMonkey(self.project, None)
        # self.developer = Developer(self.project)
        convo = self.implement_code_changes(development_plan, specs=specs)
        return convo
        template = open(os.path.join(os.path.dirname(
            __file__), 'prompts/breakdown.prompt')).read()
        prompt = self.get_prompt(template)
        dev_task = self.developer.implement_task_from_others(
            development_plan, prompt)
        output = self.implement_code_changes(dev_task)

    def get_prompt(self, template, data=None):
        from jinja2 import Environment, FileSystemLoader
        if data is None:
            data = {}
        jinja_env = Environment()
        rendered = jinja_env.from_string(
            template).render(data)
        return rendered

    def _extract_changes_from_response(self, response):
        # Parse response to extract file changes
        changes = self.postprocess_response(response)
        return changes

    def _format_function_map(self):
        return {
            'get_files': self.project.get_files,
            'save_file': self.project.save_file
        }
