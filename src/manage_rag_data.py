import os
import json
import utils
import argparse
import logging
from logging_config import setup_logging

# load the configurations file
with open("configurations.json", "r") as f:
    configurations = json.load(f)

# path of folders to the collection of docs/books and database
DOCUMENT_COLLECTION_FOLDER_RELPATH = os.path.join(
    os.path.dirname(__file__),
    configurations["document_collection_folder_relpath"])

# set up logging
setup_logging()
logger = logging.getLogger("rag_data_management_logger")


def build_databases_and_faiss_index():
    """Method to build a database that maintains the list of all docs
    and embeddings of its chunks. Also a faiss index file is built and
    saved.

    Args:
        None

    Returns:
        None
    """
    conn, cursor = utils.create_database()
    try:
        file_paths = utils.get_filepaths(DOCUMENT_COLLECTION_FOLDER_RELPATH,
                                         (".pdf",))
        utils.insert_files_metainfo_into_database(conn, cursor, file_paths)
        for path in file_paths:
            extracted_text = utils.extract_text_from_pdf(path)
            chuncked_text = utils.chunk_text(extracted_text)
            doc_id = utils.get_doc_id(path, conn, cursor)
            utils.embed_and_store_chunks(doc_id, chuncked_text, conn, cursor)
        faiss_index, mapping_data = utils.build_faiss_index(cursor)
        utils.save_faiss_index(faiss_index)
        utils.save_faiss_mapping(mapping_data, conn, cursor)
        conn.close()
        logger.info("Completed: Built all the databases and faiss index.")
    except Exception as e:
        utils.cleanup(conn)
        raise e


def update_databases_and_faiss_index():
    """Method to update the database that maintains the list of all docs
    and embeddings of its chunks when some files are deleted or added to the
    collection. Also the faiss index is rebuilt and saved.

    Args:
        None

    Returns:
        None
    """
    conn, cursor = utils.connect_to_db()
    try:
        deleted_files_paths = utils.get_list_of_deleted_files(
            DOCUMENT_COLLECTION_FOLDER_RELPATH, conn, cursor)
        if deleted_files_paths:
            utils.delete_file_metadata_in_db(deleted_files_paths,
                                             conn, cursor)
        new_files_path = utils.get_list_of_new_files(
            DOCUMENT_COLLECTION_FOLDER_RELPATH, conn, cursor)
        if new_files_path:
            utils.insert_files_metainfo_into_database(conn, cursor,
                                                      new_files_path)
            utils.delete_faiss_index(conn, cursor)
            for path in new_files_path:
                extracted_text = utils.extract_text_from_pdf(path)
                chuncked_text = utils.chunk_text(extracted_text)
                doc_id = utils.get_doc_id(path, conn, cursor)
                utils.embed_and_store_chunks(doc_id, chuncked_text, conn,
                                             cursor)
            faiss_index, mapping_data = utils.build_faiss_index(cursor)
            utils.save_faiss_index(faiss_index)
            utils.save_faiss_mapping(mapping_data, conn, cursor)
        conn.close()
        logger.info("Completed: Updated all the databases and rebuilt the "
                    "faiss index.")
    except Exception as e:
        conn.close()
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose an operation.")
    parser.add_argument("choice",
                        choices=["rebuild_all", "update_only"],
                        help="Specify the operation to be performed:"
                        " 'rebuild_all' or 'update_only'.")
    args = parser.parse_args()
    if args.choice == "rebuild_all":
        build_databases_and_faiss_index()
    elif args.choice == "update_only":
        update_databases_and_faiss_index()
    else:
        logger.info("Invalid operation requested by the user.")
