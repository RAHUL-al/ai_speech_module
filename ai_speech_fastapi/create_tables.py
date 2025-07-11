from sqlalchemy import create_engine
from models import Base
import os
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])
Base.metadata.create_all(bind=engine)
