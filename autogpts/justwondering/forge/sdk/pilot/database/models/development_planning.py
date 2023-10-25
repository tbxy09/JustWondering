from peewee import *

from .components.progress_step import ProgressStep
from playhouse.postgres_ext import BinaryJSONField


class DevelopmentPlanning(ProgressStep):
    development_plan = BinaryJSONField()

    class Meta:
        db_table = 'development_planning'
