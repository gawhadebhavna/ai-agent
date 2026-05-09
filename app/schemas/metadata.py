from pydantic import BaseModel
from typing import List, Optional


class ColumnMetadata(BaseModel):
    name: str
    type: str
    nullable: bool
    description: Optional[str] = None


class SAPMetadata(BaseModel):
    entity_name: str
    table_name: str
    schema_version: str
    description: Optional[str]
    load_type: str
    primary_keys: List[str]
    watermark_column: Optional[str]
    columns: List[ColumnMetadata]