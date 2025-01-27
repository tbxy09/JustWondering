# utils/utils.py

import os
import platform
import distro
import json
import hashlib
import re
from jinja2 import Environment, FileSystemLoader
from termcolor import colored

from forge.sdk.pilot.const.llm import MAX_QUESTIONS, END_RESPONSE
from forge.sdk.pilot.const.common import ROLES, STEPS
from forge.sdk.pilot.logger import logger


def capitalize_first_word_with_underscores(s):
    # Split the string into words based on underscores.
    words = s.split('_')

    # Capitalize the first word and leave the rest unchanged.
    words[0] = words[0].capitalize()

    # Join the words back into a string with underscores.
    capitalized_string = '_'.join(words)

    return capitalized_string


def render_prompt(path, data):
    # Create a FileSystemLoader
    file_loader = FileSystemLoader(path)

    # Create the Jinja2 environment
    env = Environment(loader=file_loader)

    # Load the template
    template = env.get_template('prompt')

    # Render the template with no variables
    content = template.render(data)

    return content


def get_prompt_components():
    # This function reads and renders all prompts inside /prompts/components and returns them in dictionary

    # Create an empty dictionary to store the file contents.
    prompts_components = {}
    data = {
        'MAX_QUESTIONS': MAX_QUESTIONS,
        'END_RESPONSE': END_RESPONSE
    }

    # Create a FileSystemLoader
    file_loader = FileSystemLoader('prompts/components')

    # Create the Jinja2 environment
    env = Environment(loader=file_loader)

    # Get the list of template names
    template_names = env.list_templates()

    # For each template, load and store its content
    for template_name in template_names:
        # Get the filename without extension as the dictionary key.
        file_key = os.path.splitext(template_name)[0]

        # Load the template and render it with no variables
        file_content = env.get_template(template_name).render(data)

        # Store the file content in the dictionary
        prompts_components[file_key] = file_content

    return prompts_components


def get_sys_message(role):
    # Create a FileSystemLoader
    file_loader = FileSystemLoader(os.path.join(
        os.path.dirname(__file__), '../prompts/system_messages'))

    # Create the Jinja2 environment
    env = Environment(loader=file_loader)

    # Load the template
    template = env.get_template(f'{role}.prompt')

    # Render the template with no variables
    content = template.render()

    return {
        "role": "system",
        "content": content
    }


def find_role_from_step(target):
    for role, values in ROLES.items():
        if target in values:
            return role

    return 'product_owner'


def get_os_info():
    os_info = {
        "OS": platform.system(),
        "OS Version": platform.version(),
        "Architecture": platform.architecture()[0],
        "Machine": platform.machine(),
        "Node": platform.node(),
        "Release": platform.release(),
    }

    if os_info["OS"] == "Linux":
        os_info["Distribution"] = ' '.join(
            distro.linux_distribution(full_distribution_name=True))
    elif os_info["OS"] == "Windows":
        os_info["Win32 Version"] = ' '.join(platform.win32_ver())
    elif os_info["OS"] == "Mac":
        os_info["Mac Version"] = platform.mac_ver()[0]

    # Convert the dictionary to a readable text format
    return array_of_objects_to_string(os_info)


def execute_step(matching_step, current_step):
    matching_step_index = STEPS.index(
        matching_step) if matching_step in STEPS else None
    current_step_index = STEPS.index(
        current_step) if current_step in STEPS else None

    return matching_step_index is not None and current_step_index is not None and current_step_index >= matching_step_index


def step_already_finished(args, step):
    args.update(step['app_data'])

    message = f"{capitalize_first_word_with_underscores(step['step'])} already done for this app_id: {args['app_id']}. Moving to next step..."
    print(colored(message, "green"))
    logger.info(message)


def generate_app_data(args):
    return {'app_id': args['app_id'], 'app_type': args['app_type']}


def array_of_objects_to_string(array):
    return '\n'.join([f'{key}: {value}' for key, value in array.items()])


def hash_data(data):
    serialized_data = json.dumps(replace_functions(
        data), sort_keys=True).encode('utf-8')
    return hashlib.sha256(serialized_data).hexdigest()


def replace_functions(obj):
    if isinstance(obj, dict):
        return {k: replace_functions(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_functions(item) for item in obj]
    elif callable(obj):
        return "function"
    else:
        return obj


def fix_json(s):
    s = s.replace('True', 'true')
    s = s.replace('False', 'false')
    # s = s.replace('`', '"')
    return fix_json_newlines(s)


def fix_json_newlines(s):
    pattern = r'("(?:\\\\n|\\.|[^"\\])*")'

    def replace_newlines(match):
        return match.group(1).replace('\n', '\\n')

    return re.sub(pattern, replace_newlines, s)


def clean_filename(filename):
    # Remove invalid characters
    cleaned_filename = re.sub(r'[<>:"/\\|?*]', '', filename)

    # Replace whitespace with underscore
    cleaned_filename = re.sub(r'\s', '_', cleaned_filename)

    return cleaned_filename
