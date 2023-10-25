import os
import forge.init_dot as init_dot

from forge.logger.debug import PathLogger
from peewee import PostgresqlDatabase

from forge.agent import ForgeAgent
from forge.sdk import LocalWorkspace
from .db import ForgeDatabase
from forge.sdk.ReactChatAgent import ReactChatAgent
from forge.sdk.CodeImp import CodeMonkeyRefactored
from forge.sdk.pilot.utils import RunShell
from forge.sdk.pilot.helpers.prompts import ZeroShotReactPrompt

database_name = os.getenv("DATABASE_STRING")
workspace = LocalWorkspace(os.getenv("AGENT_WORKSPACE"))
os.environ["AGENT_WORKSPACE"] = workspace.mkdir(task_id="", path="")
logger = PathLogger(__name__)
logger.info(
    f"=======================Agent workspace: {os.environ.get('AGENT_WORKSPACE')}")

DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Establish connection to the database
database = PostgresqlDatabase(
    DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
# check if database connection is established
try:
    if database.is_closed():
        database.connect()
except Exception as e:
    init_dot.creatdb()


database = ForgeDatabase(database_name, debug_enabled=False)
agent = ForgeAgent(database=database, workspace=workspace)

codeagent = CodeMonkeyRefactored()
pilot_agent = ReactChatAgent(
    logger=PathLogger(__name__),
    workspace=workspace,
    version="0.0.1",
    # description="elon is an experienced and visionary entrepreneur. He is able to create a startup from scratch and get a strong team to support his ideas",
    description="user is a developer. He is able to create an app from scratch and improve the app with a bunch of tools and agents",
    plugins=[
        RunShell(),
        codeagent
    ],
    target_tasks=[
        "create a new project and update the project",
        "arrange a bunch of tools and agents to do coding"],
    prompt_template=ZeroShotReactPrompt,
)
# build api server
agent.setup_agent(pilot_agent.api_task, pilot_agent.api_step,
                  pilot_agent.artifact_handler)
app = agent.get_agent_app()
