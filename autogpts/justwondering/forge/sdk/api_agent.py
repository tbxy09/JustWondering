import os
from pathlib import Path
from io import BytesIO
from typing import Annotated
from uuid import uuid4
import aiofiles
from peewee import *
from uuid import uuid4
from datetime import datetime

import uvicorn
from fastapi import APIRouter, FastAPI, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from sdk.api_db import AgentDB
from sdk.api_schema import *

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


class ServerAgent:
    def __init__(self, database: AgentDB):
        self.db = database

    def setup_agent(self, llm_task_handler, llm_step_handler, artifact_handler):
        self.llm_task_handler = llm_task_handler
        self.llm_step_handler = llm_step_handler
        self.artifact_handler = artifact_handler

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        Create a task for the agent.
        """
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
        # response = self.llm_task_handler(task_request.input)
        # # quit()
        # if response["artifact"]:
        #     await self.create_artifact(task_id=task.id)
        # return task

    async def list_tasks(self, page: int = 1, pageSize: int = 10) -> TaskListResponse:
        """
        List all tasks that the agent has created.
        """
        try:
            tasks, pagination = await self.db.list_tasks(page, pageSize)
            response = TaskListResponse(tasks=tasks, pagination=pagination)
            return response
        except Exception as e:
            raise

    async def get_task(self, task_id: str) -> Task:
        """
        Get a task by ID.
        """
        try:
            task = await self.db.get_task(task_id)
        except Exception as e:
            raise
        return task

    async def list_steps(
        self, task_id: str, page: int = 1, pageSize: int = 10
    ) -> TaskStepsListResponse:
        """
        List the IDs of all steps that the task has created.
        """
        try:
            steps, pagination = await self.db.list_steps(task_id, page, pageSize)
            response = TaskStepsListResponse(
                steps=steps, pagination=pagination)
            return response
        except Exception as e:
            raise

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        """
        Execute a step for the task.
        """
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

    async def get_step(self, task_id: str, step_id: str) -> Step:
        """
        Get a step by ID.
        """
        try:
            step = await self.db.get_step(task_id, step_id)
            return step
        except Exception as e:
            raise

    async def get_all_artifacts(self, task_id: str) -> List[Artifact]:
        """
        Get all artifacts by ID.
        """
        try:
            logger.debug("get_all_artifacts")
            # artifacts = await self.db.list_artifacts(task_id, 1, 100)
            # ret = []
            # for artifact in artifacts[0]:
            #     ret.append(self.get_artifact(task_id, artifact.artifact_id))
            # return ret
        except Exception as e:
            raise

    async def list_artifacts(
        self, task_id: str, page: int = 1, pageSize: int = 10
    ) -> TaskArtifactsListResponse:
        """
        List the artifacts that the task has created.
        """
        task_id = task_id
        logger.debug(f"list_artifacts: {task_id}")
        try:
            artifacts, pagination = await self.db.list_artifacts(
                task_id, page, pageSize
            )
            logger.debug(
                f"list_artifacts: {artifacts} with pagination {pagination} and task_id {task_id}")
            return TaskArtifactsListResponse(artifacts=artifacts, pagination=pagination)

        except Exception as e:
            raise
    # create artifact to retrieve the project plugins of llm

    async def upload_agent_task_artifacts(self,
                                          task_id: str,
                                          file: Annotated[UploadFile, File()],
                                          relative_path: Annotated[Optional[str], Form(
                                          )] = None,
                                          ) -> Artifact:
        """
        Upload an artifact for the specified task.
        """
        file_name = file.filename or str(uuid4())
        await self.db.get_task(task_id)
        relative_path = relative_path or ""
        artifact = await self.db.create_artifact(
            task_id=task_id,
            agent_created=False,
            file_name=file_name,
            relative_path=relative_path,
        )

        path = os.path.join(os.path.dirname(__file__),
                            "../../workspace", task_id, "artifacts_in")
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "../../workspace")):
            os.mkdir(os.path.join(os.path.dirname(
                __file__), "../../workspace"))
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "../../workspace", task_id)):
            os.mkdir(os.path.join(os.path.dirname(
                __file__), "../../workspace", task_id))
        if not os.path.exists(path):
            os.mkdir(os.path.join(os.path.dirname(
                __file__), "../../workspace", task_id, "artifacts_in"))
        async with aiofiles.open(os.path.join(path, file_name), "wb") as f:
            # async read chunk ~1MiB
            while content := await file.read(1024 * 1024):
                await f.write(content)
        full_path = os.path.join(path, file_name)
        (File.insert(task_id=task_id, path='', name=file_name, full_path=full_path)
         .on_conflict(
            conflict_target=[File.task_id, File.name, File.path],
            preserve=[],
            update={'name': file_name, 'path': '', 'full_path': full_path})
         .execute())
        return artifact

    async def create_artifact(
        self, task_id: str
    ) -> Artifact:
        """
        Create an artifact for the task.
        """
        try:

            artifact = await self.db.create_artifact(
                task_id=task_id,
                file_name=self.artifact_handler.get_file_name(),
                relative_path=self.artifact_handler.get_relative_path(),
                agent_created=False,
            )
        except Exception as e:
            raise
        return artifact

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
