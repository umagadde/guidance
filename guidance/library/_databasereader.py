"""SQL wrapper around SQLDatabase in langchain."""
from typing import Any, Dict, List, Tuple, Optional

from guidance.library._sql_database import SQLDatabase as LangchainSQLDatabase
from sqlalchemy import MetaData, create_engine, insert, text
from sqlalchemy.engine import Engine


class SQLDataBase(LangchainSQLDatabase):
    """SQL Database.

    Wrapper around SQLDatabase object from langchain. Offers
    some helper utilities for insertion and querying.
    See `langchain documentation <https://tinyurl.com/4we5ku8j>`_ for more details:

    Args:
        *args: Arguments to pass to langchain SQLDatabase.
        **kwargs: Keyword arguments to pass to langchain SQLDatabase.

    """

    @property
    def engine(self) -> Engine:
        """Return SQL Alchemy engine."""
        return self._engine

    @property
    def metadata_obj(self) -> MetaData:
        """Return SQL Alchemy metadata."""
        return self._metadata

    @classmethod
    def from_uri(
        cls, database_uri: str, engine_args: Optional[dict] = None, **kwargs: Any
    ) -> "SQLDataBase":
        """Construct a SQLAlchemy engine from URI."""
        _engine_args = engine_args or {}
        return cls(create_engine(database_uri, **_engine_args), **kwargs)

    def get_table_columns(self, table_name: str) -> List[Any]:
        """Get table columns."""
        return self._inspector.get_columns(table_name)

    def get_single_table_info(self, table_name: str) -> str:
        """Get table info for a single table."""
        # same logic as table_info, but with specific table names
        template = (
            "Table '{table_name}' has columns: {columns} "
            "and foreign keys: {foreign_keys}."
        )
        columns = []
        for column in self._inspector.get_columns(table_name):
            columns.append(f"{column['name']} ({str(column['type'])})")
        column_str = ", ".join(columns)
        foreign_keys = []
        for foreign_key in self._inspector.get_foreign_keys(table_name):
            foreign_keys.append(
                f"{foreign_key['constrained_columns']} -> "
                f"{foreign_key['referred_table']}.{foreign_key['referred_columns']}"
            )
        foreign_key_str = ", ".join(foreign_keys)
        table_str = template.format(
            table_name=table_name, columns=column_str, foreign_keys=foreign_key_str
        )
        return table_str

    def insert_into_table(self, table_name: str, data: dict) -> None:
        """Insert data into a table."""
        table = self._metadata.tables[table_name]
        stmt = insert(table).values(**data)
        with self._engine.connect() as connection:
            connection.execute(stmt)
            connection.commit()

    def run_sql(self, command: str) -> Tuple[str, Dict]:
        """Execute a SQL statement and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        """
        with self._engine.connect() as connection:
            cursor = connection.execute(text(command))
            if cursor.returns_rows:
                result = cursor.fetchall()
                return str(result), {"result": result}
        return "", {}

from typing import Any, List, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

"""Base reader class."""
from abc import abstractmethod
from typing import Any, List,Optional
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
"""Base schema for data structures."""
from abc import abstractmethod
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, List, Optional,Set

from dataclasses_json import DataClassJsonMixin

import uuid
def get_new_id(d: Set) -> str:
    """Get a new ID."""
    while True:
        new_id = str(uuid.uuid4())
        if new_id not in d:
            break
    return new_id

def _validate_is_flat_dict(metadata_dict: dict) -> None:
    """
    Validate that metadata dict is flat,
    and key is str, and value is one of (str, int, float, None).
    """
    for key, val in metadata_dict.items():
        if not isinstance(key, str):
            raise ValueError("Metadata key must be str!")
        if not isinstance(val, (str, int, float, type(None))):
            raise ValueError("Value must be one of (str, int, float, None)")


@dataclass
class BaseDocument(DataClassJsonMixin):
    """Base document.

    Generic abstract interfaces that captures both index structs
    as well as documents.

    """

    # TODO: consolidate fields from Document/IndexStruct into base class
    text: Optional[str] = None
    doc_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    doc_hash: Optional[str] = None

    """"
    metadata fields
    - injected as part of the text shown to LLMs as context
    - used by vector DBs for metadata filtering

    This must be a flat dictionary, 
    and only uses str keys, and (str, int, float) values.
    """
    extra_info: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Post init."""
        # assign doc_id if not set
        if self.doc_id is None:
            self.doc_id = get_new_id(set())
        if self.doc_hash is None:
            self.doc_hash = self._generate_doc_hash()

        if self.extra_info is not None:
            _validate_is_flat_dict(self.extra_info)

    def _generate_doc_hash(self) -> str:
        """Generate a hash to represent the document."""
        doc_identity = str(self.text) + str(self.extra_info)
        return sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest()

    @classmethod
    @abstractmethod
    def get_type(cls) -> str:
        """Get Document type."""

    @classmethod
    def get_types(cls) -> List[str]:
        """Get Document type."""
        # TODO: remove this method
        # a hack to preserve backwards compatibility for vector indices
        return [cls.get_type()]

    def get_text(self) -> str:
        """Get text."""
        if self.text is None:
            raise ValueError("text field not set.")
        return self.text

    def get_doc_id(self) -> str:
        """Get doc_id."""
        if self.doc_id is None:
            raise ValueError("doc_id not set.")
        return self.doc_id

    def get_doc_hash(self) -> str:
        """Get doc_hash."""
        if self.doc_hash is None:
            raise ValueError("doc_hash is not set.")
        return self.doc_hash

    @property
    def is_doc_id_none(self) -> bool:
        """Check if doc_id is None."""
        return self.doc_id is None

    @property
    def is_text_none(self) -> bool:
        """Check if text is None."""
        return self.text is None

    def get_embedding(self) -> List[float]:
        """Get embedding.

        Errors if embedding is None.

        """
        if self.embedding is None:
            raise ValueError("embedding not set.")
        return self.embedding

    @property
    def extra_info_str(self) -> Optional[str]:
        """Extra info string."""
        if self.extra_info is None:
            return None

        return "\n".join([f"{k}: {str(v)}" for k, v in self.extra_info.items()])
@dataclass
class Document(BaseDocument):
    """Generic interface for a data document.

    This document connects to data sources.

    """

    def __post_init__(self) -> None:
        """Post init."""
        super().__post_init__()
        if self.text is None:
            raise ValueError("text field not set.")

    @classmethod
    def get_type(cls) -> str:
        """Get Document type."""
        return "Document"



class BaseReader:
    """Utilities for loading data from a directory."""

    @abstractmethod
    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""

    




class DatabaseReader(BaseReader):
    """Simple Database reader.

    Concatenates each row into Document used by LlamaIndex.

    Args:
        sql_database (Optional[SQLDatabase]): SQL database to use,
            including table names to specify.
            See :ref:`Ref-Struct-Store` for more details.

        OR

        engine (Optional[Engine]): SQLAlchemy Engine object of the database connection.

        OR

        uri (Optional[str]): uri of the database connection.

        OR

        scheme (Optional[str]): scheme of the database connection.
        host (Optional[str]): host of the database connection.
        port (Optional[int]): port of the database connection.
        user (Optional[str]): user of the database connection.
        password (Optional[str]): password of the database connection.
        dbname (Optional[str]): dbname of the database connection.

    Returns:
        DatabaseReader: A DatabaseReader object.
    """

    def __init__(
        self,
        sql_database: Optional[SQLDataBase] = None,
        engine: Optional[Engine] = None,
        uri: Optional[str] = None,
        scheme: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        dbname: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize with parameters."""
        if sql_database:
            self.sql_database = sql_database
        elif engine:
            self.sql_database = SQLDataBase(engine, *args, **kwargs)
        elif uri:
            self.uri = uri
            self.sql_database = SQLDataBase.from_uri(uri, *args, **kwargs)
        elif scheme and host and port and user and password and dbname:
            uri = f"{scheme}://{user}:{password}@{host}:{port}/{dbname}"
            self.uri = uri
            self.sql_database = SQLDataBase.from_uri(uri, *args, **kwargs)
        else:
            raise ValueError(
                "You must provide either a SQLDatabase, "
                "a SQL Alchemy Engine, a valid connection URI, or a valid "
                "set of credentials."
            )

    def load_data(self, query: str) -> List[Document]:
        """Query and load data from the Database, returning a list of Documents.

        Args:
            query (str): Query parameter to filter tables and rows.

        Returns:
            List[Document]: A list of Document objects.
        """
        documents = []
        with self.sql_database.engine.connect() as connection:
            if query is None:
                raise ValueError("A query parameter is necessary to filter the data")
            else:
                result = connection.execute(text(query))

            for item in result.fetchall():
                # fetch each item
                doc_str = ", ".join([str(entry) for entry in item])
                documents.append(Document(doc_str))
        return documents