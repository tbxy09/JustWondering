from peewee import *
from .components.progress_step import ProgressStep


class ProjectDescription(ProgressStep):
    prompt = TextField()
    summary = TextField()

    class Meta:
        db_table = 'project_description'
