from forge.sdk.pilot.const.function_calls import GET_FILES, DEV_STEPS, IMPLEMENT_CHANGES, CODE_CHANGES
from forge.sdk.pilot.database.models.files import File
from forge.sdk.pilot.helpers.files import update_file
from forge.sdk.pilot.helpers.AgentConvo import AgentConvo
from forge.sdk.pilot.helpers.Agent import Agent
from forge.sdk.pilot.logger.logger import logger
from forge.logger.debug import PathLogger
import json
import os

logger = PathLogger(__name__)
# logger.addHandler(logging.FileHandler(
#     filename='./logger/debug.log', mode='w'))


class CodeMonkey(Agent):
    def __init__(self, project, developer):
        super().__init__('code_monkey', project)
        self.developer = developer

    def implement_code_changes(self, convo, code_changes_description, step_index=0, specs=None):
        if convo == None:
            convo = AgentConvo(self)
            convo.messages[-1] = {
                "role": "system",
                "content": f"""
Now, tell me all the code that needs to be written to implement this app, be careful with the escape character and make sure backslash (\) not followed by an unexpected character in the string and also make sure there is no unescaped control characters ,have it fully working and all commands that need to be run to implement this app.
!IMPORTANT!
Remember, I'm currently in an empty folder where I will start writing files that you tell me.
resolve all to do comments, and provide a fully accomplished code, and no "pass" inside a python code.
the todo comments like below, need to provide fully accomplished code:
def process_task(task):
    # Implement the logic for process task
    pass
here is the specs for detail reference:{specs}
"""}
        else:
            convo.messages = []
        files_needed = convo.send_message('development/task/request_files_for_code_changes.prompt', {
            "step_description": code_changes_description,
            "directory_tree": self.project.get_directory_tree(True),
            "step_index": step_index,
            "finished_steps": ', '.join(f"#{j}" for j in range(step_index))
        }, GET_FILES)

        changes = convo.send_message('development/implement_changes.prompt', {
            "step_description": code_changes_description,
            "step_index": step_index,
            "directory_tree": self.project.get_directory_tree(True),
            "files": self.project.get_files(files_needed),
        }, IMPLEMENT_CHANGES)
        for msg in convo.messages:
            logger.debug(json.dumps(msg, indent=4))
        path = os.path.join(os.path.dirname(
            __file__), '../../../../prompts/execute_commands.prompt')
        open(path, 'a').write(
            json.dumps(msg['role']+":"+msg['content'])+"\n")
        convo.remove_last_x_messages(1)

        if ('update_files_before_start' not in self.project.args) or (self.project.skip_until_dev_step != str(self.project.checkpoints['last_development_step'].id)):
            for file_data in changes:
                self.project.save_file(file_data)

        return convo
