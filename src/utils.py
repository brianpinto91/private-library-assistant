import os
import sqlite3
from datetime import datetime
import json
import pymupdf
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import faiss
import logging
from logging_config import setup_logging

# load the configurations file
with open("configurations.json", "r") as f:
    configurations = json.load(f)

# path of folders to the collection of docs/books and database
DOCUMENT_COLLECTION_FOLDER_RELPATH = os.path.join(
    os.path.dirname(__file__),
    configurations["document_collection_folder_relpath"])

# path of database to keep a track of docs and books
DATABASES_FOLDER_RELPATH = os.path.join(
    os.path.dirname(__file__),
    configurations["database_folder_relpath"])
DB_PATH = os.path.join(
    DATABASES_FOLDER_RELPATH,
    configurations["database"]["name"])
FAISS_INDEX_FPATH = os.path.join(
    DATABASES_FOLDER_RELPATH,
    configurations["database"]["faiss_index_fname"])

# set up logging
setup_logging()
logger = logging.getLogger("rag_data_management_logger")


def create_database() -> sqlite3.Connection:
    """Method to create a fresh database by deleting the old
    database if it exists.

    Args:
        None

    Returns:
        sqlite3.Connection: connection object.
    """
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    logger.info(f"Database created successfully: {DB_PATH}")
    create_tables(conn, cursor)
    return conn, cursor


def create_tables(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    """Method to create tables in the database to store the
    names of the documents and books in the private library and
    also to store the processed documents as chunks and embeddings.

    Args:
        conn (sqlite3.Connection): connection object.
        cursor (sqlite3.Cursor): cursor object.

    Returns:
        None
    """
    table_name_docs = configurations["database"]["table_name_documents"]
    table_name_embds = configurations["database"]["table_name_embeddings"]
    table_name_mapping = configurations["database"]["table_name_faiss_map"]
    cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name_docs}(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    filepath TEXT UNIQUE NOT NULL,
                    last_modified TEXT NOT NULL,
                    inserted_at TEXT NOT NULL);
                """)
    cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name_embds}(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES {table_name_docs}(id));
                """)
    cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name_mapping}(
                    faiss_id INTEGER PRIMARY KEY,
                    embd_id INTEGER NOT NULL,
                    FOREIGN KEY(embd_id) REFERENCES {table_name_embds}(id));
                """)
    conn.commit()
    logger.info("Tables created successfully.")


def get_filepaths(root_folder: str, file_type: tuple = (".pdf",))\
                  -> list[str]:
    """Method to get the file path of files of specified type from the provided
    list of directories.

    Args:
        root_folder (str): the root folder to search for files.
        file_types (tuple(str)): tuple of file types to search for.

    Returns:
        list(str): list of file paths.
    """
    file_paths = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(file_type):
                file_paths.append(os.path.join(root, file))
    return file_paths


def get_doc_id(doc_path: str, conn: sqlite3.Connection,
               cursor: sqlite3.Cursor) -> int:
    """Method to get the document id from the database.

    Args:
        doc_path (str): the path of the document.
        conn (sqlite3.Connection): connection object.
        cursor (sqlite3.Cursor): cursor object.

    Returns:
        int: the document id.
    """
    table_name_docs = configurations["database"]["table_name_documents"]
    cursor.execute(f"""SELECT id FROM {table_name_docs} WHERE filepath = ?;""",
                   (doc_path,))
    return cursor.fetchone()[0]


def insert_files_metainfo_into_database(conn: sqlite3.Connection,
                                        cursor: sqlite3.Cursor,
                                        file_paths: list[str]) -> None:
    """Method to insert the filenames into the database of any new/modified
    doc or book in the private library.

    Args:
        conn (sqlite3.Connection): connection object.
        cursor (sqlite3.Cursor): cursor object.
        file_paths (list(str)): list of file-paths to add the file names in the
        database.

    Returns:
        None
    """
    table_name_docs = configurations["database"]["table_name_documents"]
    for path in file_paths:
        filename = os.path.basename(path)
        last_modified = datetime.fromtimestamp(
            os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
        inserted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(f"""INSERT INTO {table_name_docs} (filename, filepath,
                       last_modified, inserted_at) VALUES (?, ?, ?, ?);""",
                       (filename, path, last_modified, inserted_at))
        conn.commit()
        logger.info(f"Inserted file-metadata in database for: {filename}")


def extract_text_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    """Method to extract the text from the documents/books.

    Args:
        conn (sqlite3.Connection): connection object.
        cursor (sqlite3.Cursor): cursor object.

    Returns:
        list[tuple[int, str]]: a list of tuples representing the page number
        and the corresponding text for all pages in the document specified by
        the path.
    """
    doc = pymupdf.open(pdf_path)
    extracted_text = []
    for page_num, content in enumerate(doc, start=1):
        extracted_text.append((page_num, content.get_text("text")))
    return extracted_text


def chunk_text(doc: list[tuple[int, str]], window_size: int = 3,
               stride: int = 2)\
               -> list[tuple[int, str]]:
    """Method to split the text into chunks of a sentences of specified
    size and stride.

    Args:
        doc (list[int, str]): a list of tuples representing the page number
        and the corresponding text representing the whole document.
        window_size (int): the number of sentences to combine to make
        a chunk.
        stride (int): the number of sentences to skip when combining
        sentences to make a chunk.

    Returns:
        list[int, str]: a list of tuples representing the page number
        and the corresponding text chunk.
    """
    chunked_text = []
    for page_num, content in doc:
        sentences = sent_tokenize(content)
        for i in range(0, len(sentences) + 1 - window_size, stride):
            chunk = " ".join(sentences[i: i + window_size])
            chunked_text.append((page_num, chunk))
    return chunked_text


def embed_and_store_chunks(doc_id: int, chunked_list: list[tuple[int, str]],
                           conn: sqlite3.Connection,
                           cursor: sqlite3.Cursor) -> None:
    """Method to embed the chunks and store them along with the page number,
    document id and the chink in the database.

    Args:
        doc_id (int): the id of the document whose data is in the chunked list
        chunked_list (tuple[int, str]): a list of tuples representing the
        page number and the corresponding text.
        conn (sqlite3.Connection): connection object.
        cursor (sqlite3.Cursor): cursor object.

    Returns:
        None
    """
    model = SentenceTransformer(
        model_name_or_path=configurations["embedding_model"]["model_name"],
        device=configurations["embedding_model"]["model_device"])
    batch_size = configurations["embedding_model"]["batch_size"]
    table_name_embs = configurations["database"]["table_name_embeddings"]
    for i in range(0, len(chunked_list), batch_size):
        batch = chunked_list[i: i + batch_size]
        batch_page_nums, batch_chunks = zip(*batch)
        batch_embeddings = model.encode(batch_chunks,
                                        normalize_embeddings=True)
        batch_doc_id = [doc_id] * len(batch_embeddings)
        batch_page_nums = list(batch_page_nums)
        batch_chunks = list(batch_chunks)
        batch_embeddings = [pickle.dumps(emb) for
                            emb in list(batch_embeddings)]
        data = tuple(zip(batch_doc_id, batch_page_nums, batch_chunks,
                         batch_embeddings))
        cursor.executemany(f"""INSERT INTO {table_name_embs} (doc_id,
                           page_number, chunk, embedding)
                           VALUES (?, ?, ?, ?);""", data)
        conn.commit()
    logger.info(f"Inserted embeddings for document id: {doc_id}")


def build_faiss_index(cursor: sqlite3.Cursor) -> tuple[faiss.IndexFlatL2,
                                                       list[tuple[int, int]]]:
    """Method to build a faiss index from the embeddings stored in the
     database.

    Args:
        cursor (sqlite3.Cursor): cursor object.

    Returns:
        tuple[faiss.IndexFlatL2, list[tuple[int, int]]]: faiss index and the
        mapping from faiss index to embedding index in the embedding table.
    """
    table_name_embds = configurations["database"]["table_name_embeddings"]
    cursor.execute(f"""SELECT id, embedding FROM {table_name_embds};""")
    rows = cursor.fetchall()
    id, embedding = zip(*rows)
    id, embedding = list(id), [pickle.loads(e) for e in embedding]
    embedding = np.array(embedding)

    # build faiss index
    embedding_dim = embedding.shape[1]
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    faiss_index.add(embedding)

    # build a new mapping table of faiss index to embeddings
    faiss_id = [i for i in range(embedding.shape[0])]
    mapping_data = list(zip(faiss_id, id))
    return faiss_index, mapping_data


def save_faiss_index(faiss_index: faiss.IndexFlatL2) -> None:
    """Method to save the faiss index to a file.

    Args:
        faiss_index (faiss.IndexFlatL2): faiss index.

    Returns:
        None
    """
    faiss.write_index(faiss_index, FAISS_INDEX_FPATH)
    logger.info(f"Saved faiss index to file: {FAISS_INDEX_FPATH}")


def save_faiss_mapping(mapping_data: list[tuple[int, int]],
                       conn: sqlite3.Connection,
                       cursor: sqlite3.Cursor) -> None:
    """Method to save the mapping of faiss index to embedding index
    in the database.

    Args:
        mapping_data (list[tuple[int, int]]): faiss index.
        conn (sqlite3.Connection): connection object.
        cursor (sqlite3.Cursor): cursor object.

    Returns:
        None
    """
    table_name_mapping = configurations["database"]["table_name_faiss_map"]
    cursor.executemany(f"""INSERT INTO {table_name_mapping} (faiss_id, embd_id)
                       VALUES (?, ?);""", mapping_data)
    conn.commit()
    logger.info("Inserted faiss mapping into the database.")


def cleanup(conn: sqlite3.Connection) -> None:
    """Method to save the mapping of faiss index to embedding index
        in the database.

    Args:
        mapping_data (list[tuple[int, int]]): faiss index.
        conn (sqlite3.Connection): connection object.

    Returns:
        None
    """
    if conn:
        conn.close()
        logger.info("Closed the database connection.")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        logger.info(f"Deleted the database file: {DB_PATH}")
    if os.path.exists(FAISS_INDEX_FPATH):
        os.remove(FAISS_INDEX_FPATH)
        logger.info(f"Deleted the faiss index file: {FAISS_INDEX_FPATH}")


def connect_to_db() -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    """Method to connect to the database.

    Args:
        None

    Returns:
        tuple[sqlite3.Connection, sqlite3.Cursor]: connection object."
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    return conn, cursor


def get_list_of_deleted_files(root_folder: str, conn: sqlite3.Connection,
                              cursor: sqlite3.Cursor,
                              file_type: tuple = (".pdf",)) -> list[str]:
    """Method to get the file path of any deleted files of specified type
    from the provided file type by checking the file-meta data in the db.

    Args:
        root_folder (str): the root folder to search for files.
        file_types (tuple(str)): tuple of file types to search for.

    Returns:
        list(str): list of file paths.
    """
    file_paths_in_root = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(file_type):
                file_paths_in_root.append(os.path.join(root, file))
    file_paths_in_root = set(file_paths_in_root)
    cursor.execute(f"""SELECT filepath FROM {
                   configurations["database"]["table_name_documents"]};""")
    rows = cursor.fetchall()
    file_paths_in_db = set([row[0] for row in rows])
    return list(file_paths_in_db - file_paths_in_root)


def delete_faiss_index(conn: sqlite3.Connection, cursor: sqlite3.Cursor)\
                       -> None:
    """Method to delete the faiss index from the db.

    Args:
        cursor (sqlite3.Cursor): cursor object.

    Returns:
        None
    """
    table_name_mapping = configurations["database"]["table_name_faiss_map"]
    cursor.execute(f"""DELETE FROM {table_name_mapping};""")
    conn.commit()
    logger.info("Emptied the faiss index mapping table.")
    if os.path.exists(FAISS_INDEX_FPATH):
        os.remove(FAISS_INDEX_FPATH)
        logger.info(f"Deleted the faiss index file: {FAISS_INDEX_FPATH}")


def delete_file_metadata_in_db(paths: list[str], conn: sqlite3.Connection,
                               cursor: sqlite3.Cursor) -> None:
    """Method to delete the metadata of any deleted files from the db.

    Args:
        paths (str): paths of the files which are no longer there.
        conn (sqlite3.Connection): connection object.
        cursor (sqlite3.Cursor): cursor object.

    Returns:
        None
    """
    table_name_docs = configurations["database"]["table_name_documents"]
    table_name_embds = configurations["database"]["table_name_embeddings"]
    placeholder_paths = ",".join("?" * len(paths))
    paths = tuple(paths)
    cursor.execute(f"""SELECT id FROM {table_name_docs} WHERE
                    filepath IN ({placeholder_paths});""", paths)
    doc_ids = cursor.fetchall()
    doc_ids = tuple([doc_id[0] for doc_id in doc_ids])
    placeholder_dos_ids = ",".join("?" * len(doc_ids))
    cursor.execute(f"""DELETE FROM {table_name_embds} WHERE
                    doc_id IN ({placeholder_dos_ids});""", doc_ids)
    conn.commit()
    cursor.execute(f"""DELETE FROM {table_name_docs} WHERE
                    filepath IN ({placeholder_paths});""", paths)
    conn.commit()
    for path in paths:
        logger.info(f"Deleted the file meta-data for: {path}")
    delete_faiss_index(conn, cursor)


def get_list_of_new_files(root_folder: str, conn: sqlite3.Connection,
                          cursor: sqlite3.Cursor,
                          file_type: tuple = (".pdf",)) -> list[str]:
    """Method to get the file path of any new files of specified type
    from the provided file type by checking the file-meta data in the db.

    Args:
        root_folder (str): the root folder to search for files.
        file_types (tuple(str)): tuple of file types to search for.

    Returns:
        list(str): list of file paths.
    """
    file_paths_in_root = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(file_type):
                file_paths_in_root.append(os.path.join(root, file))
    file_paths_in_root = set(file_paths_in_root)
    cursor.execute(f"""SELECT filepath FROM {
                   configurations["database"]["table_name_documents"]};""")
    rows = cursor.fetchall()
    file_paths_in_db = set([row[0] for row in rows])
    return list(file_paths_in_root - file_paths_in_db)
