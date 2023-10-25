from io import BytesIO
from pathlib import Path
import aiofiles
from fastapi import UploadFile

from fastapi.responses import StreamingResponse
from forge.sdk.schema import Artifact
from peewee import *
from datetime import datetime
import os
from uuid import uuid4

from forge.sdk import (
    Agent,
    AgentDB,
    ForgeLogger,
    Step,
    StepRequestBody,
    Task,
    TaskRequestBody,
    Workspace,
    ReactChatAgent,
    PromptEngine,
    chat_completion_request,
    ChromaMemStore
)
import json
import pprint

from forge.logger.debug import PathLogger

logger = PathLogger(__name__)

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


TABLES = [File]
for table in TABLES:
    if not database.table_exists(table):
        logger.debug(f"Creating table {table}")
        database.create_tables([table])
    else:
        logger.debug(f"Table {table} exists")


class NotFoundError(Exception):
    pass


class ForgeAgent(Agent):
    """
    The goal of the Forge is to take care of the boilerplate code, so you can focus on
    agent design.

    There is a great paper surveying the agent landscape: https://arxiv.org/abs/2308.11432
    Which I would highly recommend reading as it will help you understand the possabilities.

    Here is a summary of the key components of an agent:

    Anatomy of an agent:
         - Profile
         - Memory
         - Planning
         - Action

    Profile:

    Agents typically perform a task by assuming specific roles. For example, a teacher,
    a coder, a planner etc. In using the profile in the llm prompt it has been shown to
    improve the quality of the output. https://arxiv.org/abs/2305.14688

    Additionally, based on the profile selected, the agent could be configured to use a
    different llm. The possibilities are endless and the profile can be selected
    dynamically based on the task at hand.

    Memory:

    Memory is critical for the agent to accumulate experiences, self-evolve, and behave
    in a more consistent, reasonable, and effective manner. There are many approaches to
    memory. However, some thoughts: there is long term and short term or working memory.
    You may want different approaches for each. There has also been work exploring the
    idea of memory reflection, which is the ability to assess its memories and re-evaluate
    them. For example, condensing short term memories into long term memories.

    Planning:

    When humans face a complex task, they first break it down into simple subtasks and then
    solve each subtask one by one. The planning module empowers LLM-based agents with the ability
    to think and plan for solving complex tasks, which makes the agent more comprehensive,
    powerful, and reliable. The two key methods to consider are: Planning with feedback and planning
    without feedback.

    Action:

    Actions translate the agent's decisions into specific outcomes. For example, if the agent
    decides to write a file, the action would be to write the file. There are many approaches you
    could implement actions.

    The Forge has a basic module for each of these areas. However, you are free to implement your own.
    This is just a starting point.
    """

    def __init__(self, database: AgentDB, workspace: Workspace):
        """
        The database is used to store tasks, steps and artifact metadata. The workspace is used to
        store artifacts. The workspace is a directory on the file system.

        Feel free to create subclasses of the database and workspace to implement your own storage
        """
        super().__init__(database, workspace)
        self.llm_task_handler = None
        self.llm_step_handler = None
        self.artifact_handler = None

    def setup_agent(self, llm_task_handler, llm_step_handler, artifact_handler):
        self.llm_task_handler = llm_task_handler
        self.llm_step_handler = llm_step_handler
        self.artifact_handler = artifact_handler

    async def create_artifact(
        self, task_id: str, file: UploadFile, relative_path: str
    ) -> Artifact:
        file_name = file.filename
        await self.db.get_task(task_id)
        relative_path = relative_path or ""
        artifact = await self.db.create_artifact(
            task_id=task_id,
            agent_created=False,
            file_name=file_name,
            relative_path=relative_path,
        )
        artifacts_in = self.workspace.mkdir(task_id=task_id, path="artifacts_in")
        # artifacts_out = self.workspace.mkdir(task_id=task_id, "artifacts_out")
        # path = os.path.join(os.path.dirname(__file__),
        #                     "../../workspace", task_id, "artifacts_in")
        # if not os.path.exists(os.path.join(os.path.dirname(__file__), "../../workspace")):
        #     os.mkdir(os.path.join(os.path.dirname(
        #         __file__), "../../workspace"))
        # if not os.path.exists(os.path.join(os.path.dirname(__file__), "../../workspace", task_id)):
        #     os.mkdir(os.path.join(os.path.dirname(
        #         __file__), "../../workspace", task_id))
        # if not os.path.exists(path):
        #     os.mkdir(os.path.join(os.path.dirname(
        #         __file__), "../../workspace", task_id, "artifacts_in"))
        async with aiofiles.open(os.path.join(artifacts_in, file_name), "wb") as f:
            # async read chunk ~1MiB
            while content := await file.read(1024 * 1024):
                await f.write(content)
        full_path = os.path.join(artifacts_in, file_name)
        (File.insert(task_id=task_id, path='', name=file_name, full_path=full_path)
         .on_conflict(
            conflict_target=[File.task_id, File.name, File.path],
            preserve=[],
            update={'name': file_name, 'path': '', 'full_path': full_path})
         .execute())
        return artifact

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to create
        a task.

        We are hooking into function to add a custom log message. Though you can do anything you
        want here.
        """
        # task = await super().create_task(task_request)
        # LOG.info(
        #     f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
        # )
        if not task_request.input:
            raise Exception("No task prompt")
        task = await self.db.create_task(
            input=task_request.input,
            additional_input=task_request.additional_input,
        )
        step = await self.db.create_step(task.task_id,
                                         input=StepRequestBody(
                                             name="run", input=task.input),
                                         is_last=False)
        logger.debug(f"Task {task.task_id} created")
        # step = self.execute_step(task.task_id, step.input)
        return step
        return task

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        """
        For a tutorial on how to add your own logic please see the offical tutorial series:
        https://aiedge.medium.com/autogpt-forge-e3de53cc58ec

        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to execute
        a step.

        The task that is created contains an input string, for the benchmarks this is the task
        the agent has been asked to solve and additional input, which is a dictionary and
        could contain anything.

        If you want to get the task use:

        ```
        task = await self.db.get_task(task_id)
        ```

        The step request body is essentially the same as the task request and contains an input
        string, for the benchmarks this is the task the agent has been asked to solve and
        additional input, which is a dictionary and could contain anything.

        You need to implement logic that will take in this step input and output the completed step
        as a step object. You can do everything in a single step or you can break it down into
        multiple steps. Returning a request to continue in the step output, the user can then decide
        if they want the agent to continue or not.
        """
        # # An example that
        # step = await self.db.create_step(
        #     task_id=task_id, input=step_request, is_last=True
        # )

        # self.workspace.write(task_id=task_id, path="output.txt", data=b"Washington D.C")

        # await self.db.create_artifact(
        #     task_id=task_id,
        #     step_id=step.step_id,
        #     file_name="output.txt",
        #     relative_path="",
        #     agent_created=True,
        # )

        # step.output = "Washington D.C"

        # LOG.info(f"\t✅ Final Step completed: {step.step_id}. \n" +
        #          f"Output should be placeholder text Washington D.C. You'll need to \n" +
        #          f"modify execute_step to include LLM behavior. Follow the tutorial " +
        #          f"if confused. ")
        task = await self.db.get_task(task_id)
        if not task:
            raise Exception("No task to execute")
        response = self.llm_task_handler(task.task_id, task.input)
        path = response['path']
        logger.debug(f"Artifact path: {path['parent']}, {path['files']}")
        step = await self.db.create_step(
            task_id=task.task_id,
            input=StepRequestBody(name="ok", input=task.input),
            is_last=True
        )
        logger.debug(f"Step {step.step_id} created")
        for file in path['files']:
            path_obj = Path('./' + path['parent'] + '/' + file)
            relative_path = str(path_obj.parent)
            relative_path = "" if relative_path == "." else relative_path
            logger.debug(
                f"relative path: {relative_path}, file_name: {path_obj.name}")
            await self.db.create_artifact(
                task_id=task_id,
                step_id=step.step_id,
                relative_path=relative_path,
                file_name=path_obj.name
            )
            logger.debug(
                f"{file} created with task {task.task_id} and step {step.step_id}")
        return step

    async def get_artifact(self, task_id: str, artifact_id: str) -> Artifact:
        """
        Get an artifact by ID.
        """
        try:
            artifact = await self.db.get_artifact(artifact_id)
            if artifact.file_name not in artifact.relative_path:
                file_path = os.path.join(
                    artifact.relative_path, artifact.file_name)
            else:
                file_path = artifact.relative_path
            # retrieved_artifact = self.workspace.read(
            #     task_id=task_id, path=file_path)
            file_path = Path('./' + file_path)
            parent = str(file_path.parent)
            parent = "" if parent == "." else parent
            name = parent + '/' + file_path.name if parent else file_path.name
            logger.debug(f"file_path: {file_path} and task_id: {task_id}")
            ret = File.select().where((File.task_id == task_id) & (
                File.name == name))
            if len(ret) == 0:
                raise NotFoundError
            else:
                file = ret[0]
            logger.debug(f"file: {file.full_path}")
            retrieved_artifact = open(file.full_path, "r").read()
            logger.debug(retrieved_artifact)

        except NotFoundError as e:
            raise
        except FileNotFoundError as e:
            raise
        except Exception as e:
            raise

        return StreamingResponse(
            BytesIO(retrieved_artifact.encode()),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={artifact.file_name}"
            },
        )
