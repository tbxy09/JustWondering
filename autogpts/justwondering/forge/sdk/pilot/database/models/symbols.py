# database table Symbols
from peewee import *
from forge.sdk.pilot.database.models.components.base_models import BaseModel
from forge.sdk.pilot.database.models.development_steps import DevelopmentSteps
from forge.sdk.pilot.database.models.app import App
from forge.sdk.pilot.database.models.files import File


class Symbols(BaseModel):
    app = ForeignKeyField(App, on_delete='CASCADE')
    file = ForeignKeyField(File, on_delete='CASCADE', null=True)
    content = TextField()

    class Meta:
        db_table = 'symbols'
        indexes = (
            (('file'), True),
        )
