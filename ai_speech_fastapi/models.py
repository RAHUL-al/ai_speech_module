from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Essay(Base):
    __tablename__ = "essays"

    id = Column(Integer, primary_key=True, index=True)
    student_class = Column(String(50), nullable=False)
    accent = Column(String(50), nullable=False)
    topic = Column(String(100), nullable=False)
    mood = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    user = relationship("User", back_populates="essays")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password = Column(String(200), nullable=False)

    essays = relationship("Essay", back_populates="user", cascade="all, delete-orphan")
