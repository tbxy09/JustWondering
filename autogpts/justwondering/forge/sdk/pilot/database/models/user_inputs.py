from peewee import *

from .components.base_models import BaseModel
from .app import App


class UserInputs(BaseModel):
    id = AutoField()
    app = ForeignKeyField(App, on_delete='CASCADE')
    hash_id = CharField(null=False)
    query = TextField(null=True)
    user_input = TextField(null=True)
    previous_step = ForeignKeyField('self', null=True, column_name='previous_step')

    class Meta:
        db_table = 'user_inputs'
        indexes = (
            (('app', 'hash_id'), True),
        )