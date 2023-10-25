# create Project table in database

from peewee import *
from .components.base_models import BaseModel
from .development_steps import DevelopmentSteps
from .app import App


class ProjectM(BaseModel):
    id = AutoField()
    uid = CharField()
    name = CharField()
    description = TextField(null=True)
    github_url = TextField(null=True)
    conversation = TextField(null=True)

    class Meta:
        indexes = (
            (('uid', 'name'), True),
        )