from peewee import *

from .components.base_models import BaseModel
from .user import User


class App(BaseModel):
    user = ForeignKeyField(User, backref='apps')
    app_type = CharField(null=True)
    name = CharField(null=True)
    status = CharField(default='started')