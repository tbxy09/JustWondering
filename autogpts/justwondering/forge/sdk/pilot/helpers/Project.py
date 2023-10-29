import os

from termcolor import colored
from forge.sdk.pilot.const.common import IGNORE_FOLDERS
from peewee import *
from uuid import uuid4
from datetime import datetime

from forge.sdk.pilot.database.models.app import App
# from forge.sdk.pilot.database.database import get_app, delete_unconnected_steps_from, delete_all_app_development_data
from forge.sdk.pilot.utils.questionary import styled_text
from forge.sdk.pilot.helpers.files import get_files_content, clear_directory, update_file
from forge.sdk.pilot.helpers.cli import build_directory_tree
from forge.sdk.pilot.helpers.agents.TechLead import TechLead
from forge.sdk.pilot.helpers.agents.Developer import Developer
from forge.sdk.pilot.helpers.agents.Architect import Architect
from forge.sdk.pilot.helpers.agents.ProductOwner import ProductOwner
import forge.init_dot as init_dot
from forge.sdk.pilot.utils.files import get_parent_folder
# TODO move to a separate file
# https://github.com/Significant-Gravitas/Auto-GPT
# https://github.com/Significant-Gravitas/Auto-GPT-Benchmarks
# https://microsoft.github.io/promptflow/index.html
# https://github.com/microsoft/promptflow

print(
    f"DB_NAME, DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, {os.getenv('DB_NAME')}, {os.getenv('DB_HOST')}, {os.getenv('DB_PORT')}, {os.getenv('DB_USER')}, {os.getenv('DB_PASSWORD')}")
# Establish connection to the database
database = PostgresqlDatabase(
    "devdb",
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)
try:
    if database.is_closed():
        database.connect()
except Exception as e:
    print(e)
    init_dot.creatdb()


class BaseModel(Model):
    id = UUIDField(primary_key=True, default=uuid4)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    class Meta:
        database = database


class File(BaseModel):
    task_id = CharField(null=True)
    name = CharField()
    path = CharField()
    full_path = CharField()

    class Meta:
        db_table = 'file'
        indexes = (
            (('task_id', 'name', 'path'), True),
        )


class FileSnapshot(BaseModel):
    id = AutoField()
    file = ForeignKeyField(File, on_delete='CASCADE', null=True)
    content = TextField()

    class Meta:
        db_table = 'file_snapshot'
        indexes = (
            (('file'), True),
        )


TABLES = [File]
for table in TABLES:
    if not database.table_exists(table):
        database.create_tables([table])


class Project:
    def __init__(self, args, name=None, description=None, user_stories=None, user_tasks=None, architecture=None,
                 development_plan=None, current_step=None):
        self.args = args
        self.llm_req_num = 0
        self.command_runs_count = 0
        self.user_inputs_count = 0
        self.checkpoints = {
            'last_user_input': None,
            'last_command_run': None,
            'last_development_step': None,
        }
        # TODO make flexible
        self.root_path = ''
        self.taskid = ''
        self.skip_until_dev_step = None
        self.skip_steps = None
        # self.restore_files({dev_step_id_to_start_from})

        if current_step is not None:
            self.current_step = current_step
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if user_stories is not None:
            self.user_stories = user_stories
        if user_tasks is not None:
            self.user_tasks = user_tasks
        if architecture is not None:
            self.architecture = architecture
        # if development_plan is not None:
        #     self.development_plan = development_plan

    def start(self):
        self.project_manager = ProductOwner(self)
        self.project_manager.get_project_description()
        self.user_stories = self.project_manager.get_user_stories()
        # self.user_tasks = self.project_manager.get_user_tasks()

        self.architect = Architect(self)
        self.architecture = self.architect.get_architecture()

        self.developer = Developer(self)
        self.developer.set_up_environment()

        self.tech_lead = TechLead(self)
        self.development_plan = self.tech_lead.create_development_plan()
        self.developer.start_coding()

    def get_directory_tree(self, with_descriptions=False):
        files = {}
        if with_descriptions and False:
            files = File.select().where(File.app_id == self.args['app_id'])
            files = {snapshot.name: snapshot for snapshot in files}
        return build_directory_tree(self.root_path + '/', ignore=IGNORE_FOLDERS, files=files, add_descriptions=False)

    def get_test_directory_tree(self):
        # TODO remove hardcoded path
        return build_directory_tree(self.root_path + '/tests', ignore=IGNORE_FOLDERS)

    def get_files_list(self, task_id):
        files = File.select().where(File.task_id == task_id)
        return [os.path.join(file.path, file.name) for file in files]

    def get_all_coded_files(self, task_id):
        files = File.select().where(File.task_id == task_id)
        files = self.get_files([file.path + '/' + file.name for file in files])
        return files

    def get_files(self, files):
        files_with_content = []
        for file in files:
            # TODO this is a hack, fix it
            try:
                relative_path, full_path = self.get_full_file_path('', file)
                file_content = open(full_path, 'r').read()
            except:
                file_content = ''

            files_with_content.append({
                "path": file,
                "content": file_content
            })
        return files_with_content

    def save_file(self, data):
        # TODO fix this in prompts
        if ' ' in data['name'] or '.' not in data['name']:
            data['name'] = data['path'].rsplit('/', 1)[1]

        data['path'], data['full_path'] = self.get_full_file_path(
            data['path'], data['name'])
        update_file(data['full_path'], data['content'])
        (File.insert(task_id=self.task_id, path=data['path'], name=data['name'], full_path=data['full_path'])
            .on_conflict(
                conflict_target=[File.task_id, File.name, File.path],
                preserve=[],
                update={'name': data['name'], 'path': data['path'], 'full_path': data['full_path']})
            .execute())

    def get_full_file_path(self, file_path, file_name):
        file_path = file_path.replace('./', '', 1)
        file_path = file_path.rsplit(file_name, 1)[0]

        if file_path.endswith('/'):
            file_path = file_path.rstrip('/')

        if file_name.startswith('/'):
            file_name = file_name[1:]

        if not file_path.startswith('/') and file_path != '':
            file_path = '/' + file_path

        if file_name != '':
            file_name = '/' + file_name

        return (file_path, self.root_path + file_path + file_name)

    def save_files_snapshot(self, development_step_id):
        files = get_files_content(self.root_path, ignore=IGNORE_FOLDERS)
        development_step, created = DevelopmentSteps.get_or_create(
            id=development_step_id)

        for file in files:
            print(
                colored(f'Saving file {file["path"] + "/" + file["name"]}', 'light_cyan'))
            # TODO this can be optimized so we don't go to the db each time
            file_in_db, created = File.get_or_create(
                app=self.app,
                name=file['name'],
                path=file['path'],
                full_path=file['full_path'],
            )

            file_snapshot, created = FileSnapshot.get_or_create(
                app=self.app,
                development_step=development_step,
                file=file_in_db,
                defaults={'content': file.get('content', '')}
            )
            file_snapshot.content = content = file['content']
            file_snapshot.save()

    def restore_files(self, development_step_id):
        development_step = DevelopmentSteps.get(
            DevelopmentSteps.id == development_step_id)
        file_snapshots = FileSnapshot.select().where(
            FileSnapshot.development_step == development_step)

        clear_directory(self.root_path, IGNORE_FOLDERS)
        for file_snapshot in file_snapshots:
            update_file(file_snapshot.file.full_path, file_snapshot.content)

    def get_plugins(self):
        # wrap it as a tool plugin
        return {
            'save_files': self.save_files,
            'get_files': self.get_files,
            'get_directory_tree': self.get_directory_tree,
            'get_test_directory_tree': self.get_test_directory_tree,
            'get_all_coded_files': self.get_all_coded_files,
            'get_full_file_path': self.get_full_file_path,
        }

    def delete_all_steps_except_current_branch(self):
        delete_unconnected_steps_from(
            self.checkpoints['last_development_step'], 'previous_step')
        delete_unconnected_steps_from(
            self.checkpoints['last_command_run'], 'previous_step')
        delete_unconnected_steps_from(
            self.checkpoints['last_user_input'], 'previous_step')

    def ask_for_human_intervention(self, message, description=None, cbs={}):
        print(colored(message, "yellow", attrs=['bold']))
        if description is not None:
            print(description)
        answer = ''
        while answer != 'continue':
            answer = styled_text(
                self,
                'If something is wrong, tell me or type "continue" to continue.',
            )

            if answer in cbs:
                return cbs[answer]()
            elif answer != '':
                return answer
