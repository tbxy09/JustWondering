from forge.sdk.pilot.utils.utils import step_already_finished
from forge.sdk.pilot.helpers.Agent import Agent
import json
from termcolor import colored
from forge.sdk.pilot.const.function_calls import ARCHITECTURE

from forge.sdk.pilot.utils.utils import execute_step, find_role_from_step, generate_app_data
from forge.sdk.pilot.database.database import save_progress, get_progress_steps
from forge.sdk.pilot.logger.logger import logger
from forge.sdk.pilot.helpers.prompts import get_additional_info_from_user
from forge.sdk.pilot.helpers.AgentConvo import AgentConvo


class Architect(Agent):
    def __init__(self, project):
        super().__init__('architect', project)
        self.convo_architecture = None

    def get_architecture(self):
        self.project.current_step = 'architecture'
        self.convo_architecture = AgentConvo(self)

        # If this app_id already did this step, just get all data from DB and don't ask user again
        step = get_progress_steps(
            self.project.args['app_id'], self.project.current_step)
        if step and not execute_step(self.project.args['step'], self.project.current_step):
            step_already_finished(self.project.args, step)
            return step['architecture']

        # ARCHITECTURE
        print(colored(f"Planning project architecture...\n",
              "green", attrs=['bold']))
        logger.info(f"Planning project architecture...")

        architecture = self.convo_architecture.send_message('architecture/technologies.prompt',
                                                            {'name': self.project.args['name'],
                                                             'prompt': self.project.project_description,
                                                             'user_stories': self.project.user_stories,
                                                             #  'user_tasks': self.project.user_tasks,
                                                             'app_type': self.project.args['app_type']}, ARCHITECTURE)

        if self.project.args.get('advanced', False):
            architecture = get_additional_info_from_user(
                self.project, architecture, 'architect')

        logger.info(f"Final architecture: {architecture}")

        save_progress(self.project.args['app_id'], self.project.current_step, {
            "messages": self.convo_architecture.messages,
            "architecture": architecture,
            "app_data": generate_app_data(self.project.args)
        })

        return architecture
        # ARCHITECTURE END
