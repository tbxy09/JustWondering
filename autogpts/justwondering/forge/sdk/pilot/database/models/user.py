from peewee import *

# from forge.sdk.pilot.database.models.components.base_models import BaseModel

from .components.base_models import BaseModel


class User(BaseModel):
    email = CharField(unique=True)
    password = CharField()
