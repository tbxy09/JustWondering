from peewee import *

from .components.base_models import BaseModel
from .development_steps import DevelopmentSteps
from .app import App

class File(BaseModel):
    id = AutoField()
    app = ForeignKeyField(App, on_delete='CASCADE')
    name = CharField()
    path = CharField()
    full_path = CharField()
    description = TextField(null=True)

    class Meta:
        # database = db
        indexes = (
            (('app', 'name', 'path'), True),
        )