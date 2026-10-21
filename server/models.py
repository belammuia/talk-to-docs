from extensions import db
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import mapped_column


class Chunks(db.Model):
    __tablename__ = "chunks"

    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    page = db.Column(db.Integer,nullable=False)
    embedding = mapped_column(Vector(384))