# main.py
from __future__ import print_function, unicode_literals
from forge.sdk.pilot.database.database import database_exists, create_database, tables_exist, create_tables
from forge.sdk.pilot.logger.logger import logger
from forge.sdk.pilot.utils.arguments import get_arguments
from forge.sdk.pilot.helpers.Project import Project

from dotenv import load_dotenv
load_dotenv()


def init():
    # Check if the "euclid" database exists, if not, create it
    if not database_exists():
        create_database()

    # Check if the tables exist, if not, create them
    if not tables_exist():
        create_tables()

    arguments = get_arguments()

    logger.info(f"Starting with args: {arguments}")

    return arguments


if __name__ == "__main__":
    args = init()

    # TODO get checkpoint from forge.sdk.pilot.database and fill the project with it
    project = Project(args)
    project.start()
