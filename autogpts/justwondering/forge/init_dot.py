import os
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import psycopg2
from forge.logger.debug import PathLogger
from dotenv import load_dotenv
logger = PathLogger(__name__)
load_dotenv('.env')
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
logger.debug(
    f"DB_NAME, DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, {DB_NAME}, {DB_HOST}, {DB_PORT}, {DB_USER}, {DB_PASSWORD}")

# add check if database exists


def creatdb():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER,
                            password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute('CREATE DATABASE devdb;')
    logger.info("Database created successfully")
    cur.close()
    conn.close()
