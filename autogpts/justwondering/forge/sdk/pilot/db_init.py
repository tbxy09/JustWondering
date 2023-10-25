from forge.sdk.pilot.database.database import create_tables, drop_tables
from dotenv import load_dotenv
load_dotenv()

drop_tables()
create_tables()
