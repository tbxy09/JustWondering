#create ProjectFiles table in database

from peewee import *
from .components.base_models import BaseModel
from .projects import ProjectM
from .app import App

class ProjectFiles(BaseModel):
    id = AutoField()
    project = ForeignKeyField(ProjectM, on_delete='CASCADE')
    directory = CharField(null=True)
    name = CharField()
    file_path = CharField()
    description = TextField(null=True)

    class Meta:
        indexes = (
            (('id', 'file_path'), True),
        )