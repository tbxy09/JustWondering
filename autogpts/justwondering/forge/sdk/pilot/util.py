from .utils.utils import array_of_objects_to_string
from .utils.repo_api import get_repo_json, get_local_json, GET_REPO_FILES
from .helpers.Project import Project
from .utils.basetool import BaseTool
from .utils.base_agent import BaseAgent
from typing import List, Union, Optional, Dict, Callable
from forge.sdk.pilot.database.models.projectFiles import ProjectFiles
from forge.sdk.pilot.database.models.projects import ProjectM
from forge.sdk.pilot.database.models.symbols import Symbols
from .utils.llm_connection import create_gpt_chat_completion, get_prompt


def return_array_from_prompt(name_plural, name_singular, return_var_name):
    return {
        'name': f'process_{name_plural.replace(" ", "_")}',
        'description': f" process list of {name_plural}",
        'parameters': {
            'type': 'object',
            "properties": {
                f"{return_var_name}": {
                    "type": "array",
                    "description": f"List of {name_plural} that are created in a list.",
                    "items": {
                        "type": "string",
                        "description": f"{name_singular}"
                    },
                },
            },
            "required": [return_var_name],
        },
    }


def format_plugins_schema(plugins: Union[BaseTool, BaseAgent]):
    schema = return_array_from_prompt(
        plugins[0].name, plugins[0].description, plugins[0].args_schema.schema()['properties'].keys()[0])
    return schema


def format_multi_plugins_schema(plugins: List[Union[BaseTool, BaseAgent]]):
    """Format list of tools into the open AI function API.

    :param plugins: List of tools to be formatted.
    :type plugins: List[Union[BaseTool, BaseAgent]]
    :return: Formatted tools.
    :rtype: Dict
    """
    schema = {
        'type': 'object',
        'properties': {
            "tools": {
                'type': 'array',
                'description': 'List of tools to be run.',
                'items': {
                    'type': 'object',
                    'description': 'Tool to be run.',
                    'properties': {
                        'name': {
                            'type': 'string',
                            'enum': [plugin.name for plugin in plugins],
                            'description': 'Name of the tool.',
                        },
                        # plugins[0].args_schema.schema(),
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


def format_plugin_schema(plugin: Union[BaseTool, BaseAgent]):
    """Format tool into the open AI function API.

    :param plugin: Tool to be formatted.
    :type plugin: Union[BaseTool, BaseAgent]
    :return: Formatted tool.
    :rtype: Dict
    """
    if isinstance(plugin, Union[BaseTool, BaseAgent]):
        if plugin.args_schema:
            parameters = plugin.args_schema.schema()
        else:
            parameters = {
                # This is a hack to get around the fact that some tools
                # do not expose an args_schema, and expect an argument
                # which is a string.
                # And Open AI does not support an array type for the
                # parameters.
                "properties": {
                    "__arg1": {"title": "__arg1", "type": "string"},
                },
                "required": ["__arg1"],
                "type": "object",
            }

        return {
            "name": plugin.name,
            "description": plugin.description,
            "parameters": parameters,
        }
    else:
        parameters = plugin.args_schema.schema()
        return {
            "name": plugin.name,
            "description": plugin.description,
            "parameters": parameters,
        }


class FileSearchAgent:
    def __init__(self, bash_tool: BaseTool):
        self.bash_tool = bash_tool

    def search_files(self, keyword: str, directory: str):
        command = f"find {directory} -type f -name '*{keyword}*'"
        result = self.bash_tool.run(command)
        return result

    def tree(self, directory: str):
        command = f"tree {directory}"
        result = self.bash_tool.run(command)
        # return an json of tree
        return result


class GitUrl():
    # with method create name using github_url.split('/')[-1] + "_" + github_url.split('/')[-2]
    # hash value of github_url
    def __init__(self, github_url):
        self.value = github_url
        # gen project id from github url
        self.id = hash(github_url)
        self.name = github_url.split('/')[-1] + "_" + github_url.split('/')[-2]


class LocalPath():
    def __init__(self, path):
        self.value = path
        self.id = hash(path)
        self.name = '_'.join(path.split('/')[:])


class GitProject(Project):
    def __init__(self, args, path: Union[GitUrl, LocalPath]):
        super().__init__(args)
        print(path)
        self.github_url = path.value
        # gen project id from github url
        self.project_id = path.id
        self.project_name = path.name
        ret = self.check_if_schema_exists()
        if ret[0]:
            self.project_files = ret[1]
        elif isinstance(path, GitUrl):
            self.project_files = get_repo_json(path.value)
        else:
            self.project_files = get_local_json(path.value)
        self.save_project()
        self.save_project_files()

    def save_project(self):
        # Save project to DB
        projectm, created = ProjectM.get_or_create(
            uid=self.project_id,
            name=self.project_name,
            github_url=self.github_url,
        )

    def check_if_schema_exists(self):
        import os
        import json
        if not os.path.exists('schema.json'):
            return (False, None)
        with open('schema.json') as f:
            schema = json.load(f)
        # check the project cached in schema is the same as the current project if not return false
        project_outline = schema['System'].get(
            'outputFormat', {}).get('projectOutline', {})
        # project_files = schema.get('outputFormat', {}).get('projectFiles', {})
        repo_owner = self.github_url.split('/')[-2]
        repo_name = self.github_url.split('/')[-1]
        print(repo_name, repo_owner)
        if project_outline.get('repositoryUrl') == f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/.github?ref=main":
            return (True, schema)
        else:
            return (False, None)

    def save_project_files(self):
        # Parse project files
        for file in self.project_files["System"]["outputFormat"]["projectFiles"]["files"]:
            print(file['name'], file['path'])
            projectfilesM, created = ProjectFiles.get_or_create(
                name=file['name'],
                file_path=file['path'],
                project=ProjectM.get(uid=self.project_id),
            )

        # # Extract symbols
        # symbols = []
        # for file in self.project_files:
        #   symbols.extend(self.extract_symbols(file['content']))

        # # Save symbols
        # for symbol in symbols:
        #   Symbols.create(name=symbol)

    def get_project_files(self):
        return ProjectFiles.select()

    def query_project_files(self, query):
        # Search ProjectFiles for query
        return ProjectFiles.select().where(ProjectFiles.content ** query)

    def query_symbols_from_project_files(self, symbol):
        return Symbols.select().where(Symbols.name ** symbol)

    def extract_symbols(self, content):
        pass

    def extract_techs(self, project_files):
        pass

    def gen_technologies(self):
        return
        # Extract techs like imports
        techs = self.extract_techs(self.project_files)
        # Save to DB
        for tech in techs:
            Technologies.create(name=tech)
        self.technologies = techs

    # Other methods for splitting files, efrom gentopia.agent import BaseAgent


class AgentConvo:
    def __init__(self, agent, system_message):
        self.agent = agent
        # self.messages = [{"role":"system","content":"You are an AI assistant that helps people reading code."}]
        self.messages = [{"role": "system", "content": system_message}]
    # copy the postprocess_response from AgentConvo.py

    def remove_last_x_messages(self, x):
        self.messages = self.messages[:-x]

    def postprocess_response(self, response, function_calls):
        if function_calls is not None and 'function_calls' in function_calls:
            if 'to_message' in function_calls:
                return function_calls['to_message'](response)
            elif 'to_json' in function_calls:
                return function_calls['to_json'](response)
            elif 'to_json_array' in function_calls:
                return function_calls['to_json_array'](response)
            else:
                return response
        else:
            if type(response) == tuple:
                return response[0]
            else:
                return response

    # copy construct_and_add_message_from_prompt from AgentConvo.py
    def construct_and_add_message_from_prompt(self, prompt_path, prompt_data):
        # print("construct_and_add_message_from_prompt")
        # return
        if prompt_path is None:
            prompt_path = self.agent.get_prompt_path()
        if prompt_data is None:
            prompt_data = self.agent.get_prompt_data()
        prompt = get_prompt(prompt_path, prompt_data)
        self.messages.append({"role": self.agent.role, "content": prompt})

    def toConversationHistory(self):
        history = []
        for msg in self.messages:
            history.append(f"{msg['role']}: {msg['content']}")
        # save to clipboard
        pyperclip.copy('\n'.join(history))
        return '\n'.join(history)

    def send_message(self, prompt_path=None, prompt_data=None, function_calls=None):
        self.construct_and_add_message_from_prompt(prompt_path, prompt_data)
        if function_calls is not None and 'function_calls' in function_calls:
            self.messages[-1]['content'] += '\nMAKE SURE THAT YOU RESPOND WITH A CORRECT JSON FORMAT!!!'
        response = create_gpt_chat_completion(
            self.messages, None, function_calls=function_calls)

        # TODO handle errors from OpenAI
        if response == {}:
            raise Exception("OpenAI API error happened.")

        response = self.postprocess_response(response, function_calls)

        # TODO remove this once the database is set up properly
        message_content = response[0] if type(response) == tuple else response
        if isinstance(message_content, list):
            if 'to_message' in function_calls:
                string_response = function_calls['to_message'](message_content)
            elif len(message_content) > 0 and isinstance(message_content[0], dict):
                string_response = [
                    f'#{i}\n' + array_of_objects_to_string(d)
                    for i, d in enumerate(message_content)
                ]
            else:
                string_response = ['- ' + r for r in message_content]

            message_content = '\n'.join(string_response)
        # TODO END

        # TODO we need to specify the response when there is a function called
        # TODO maybe we can have a specific function that creates the GPT response from the function call
        self.messages.append({"role": "assistant", "content": message_content})
        # self.log_message(message_content)

        return response


def save_prompt_to_file(prompt_path, prompt_content):
    # Save the new task prompt to a file
    with open(prompt_path, "w") as f:
        f.write(prompt_content)
    f.close()
