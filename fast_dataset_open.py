import os
import sqlite3
import hashlib
import pandas as pd

# If you want to read RData files, install and import pyreadr:
#   pip install pyreadr
import pyreadr


# You can set these as global variables or pass them as arguments
CACHE_FOLDER = "cache_data"
CACHE_DB_PATH = os.path.join(CACHE_FOLDER, "cache_info.sqlite")

# Ensure our cache folder exists
os.makedirs(CACHE_FOLDER, exist_ok=True)


def _init_cache_db():
    """
    Initialize the cache database if it doesn't exist.
    We'll store:
      - file_list: a string representing the list of files
      - cache_path: path to the corresponding cached dataframe
    """
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS cache_info (
            file_list TEXT PRIMARY KEY,
            cache_path TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def _get_cache_entry(file_list_str):
    """
    Return the path to the cached file if it exists in DB, otherwise None.
    """
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT cache_path FROM cache_info WHERE file_list = ?",
        (file_list_str,)
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0]
    return None


def _store_cache_entry(file_list_str, cache_path):
    """
    Store or update the cache path entry for this set of files.
    """
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO cache_info (file_list, cache_path)
        VALUES (?, ?)
        """,
        (file_list_str, cache_path)
    )
    conn.commit()
    conn.close()


def _create_file_list_key(paths):
    """
    Create a canonical, case-insensitive representation of the file list.
    Sort them to avoid issues with ordering. For example:
      ["d.Rdata","d2.csv"] => "d.rdata|d2.csv"
    """
    # Normalize paths (sort, lower, absolute, or any scheme you like)
    normalized = []
    for p in paths:
        # For maximum consistency, consider using absolute paths
        # absolute_p = os.path.abspath(p)
        # But here we'll just lower and trim:
        normalized.append(os.path.normpath(p).lower())
    normalized.sort()
    file_list_str = "|".join(normalized)
    return file_list_str


def _read_dataset(path):
    """
    Reads a single file path, returning a pandas DataFrame.
    Supports CSV or RData for demonstration.
    """
    lower_path = path.lower()
    if lower_path.endswith(".csv"):
        df = pd.read_csv(path)
    elif lower_path.endswith(".rdata") or lower_path.endswith(".rda"):
        # Using pyreadr to load R data
        result = pyreadr.read_r(path)
        # pyreadr.read_r returns a dict {objectName -> RDataFrame}
        # We'll pick the first one
        if len(result.keys()) == 1:
            df = next(iter(result.values()))
        else:
            # You could handle multiple objects from the RData here
            # or raise an error if you expect only one object
            df = next(iter(result.values()))
    else:
        raise ValueError(f"Unsupported file format: {path}")
    return df


def open_with_cache(*paths, ignore_cache=False):
    """
    Reads one or more dataset paths (CSV or RData), concatenates them,
    returns a single pandas DataFrame. A cache is created on disk (in
    CACHE_FOLDER) in a 'faster to load' format (e.g., pickle). A small
    SQLite database tracks which set(s) of file paths map to which cache file.

    :param paths: One or more file paths (strings) to be concatenated.
    :param ignore_cache: If True, forces re-reading from original source
                         and overwriting the cache.
    :return: Concatenated pandas DataFrame
    """
    # Ensure the DB is initialized
    _init_cache_db()

    # Create a canonical key for the *group* of files requested
    file_list_str = _create_file_list_key(paths)

    # Lookup existing cache in DB
    existing_cache = _get_cache_entry(file_list_str)

    if (not ignore_cache) and existing_cache and os.path.exists(existing_cache):
        # Cache exists and user is not ignoring it => load from cache
        print(f"Loading from cache: {existing_cache}")
        df = pd.read_pickle(existing_cache)
        return df

    # Otherwise, read the datasets from source, concatenate, and create the cache
    dfs = []
    for path in paths:
        print(f"Reading file: {path}")
        dfs.append(_read_dataset(path))

    if len(dfs) == 1:
        combined_df = dfs[0]
    else:
        combined_df = pd.concat(dfs, ignore_index=True)

    # Compute a stable cache file name (e.g., use hash of the file list)
    hash_val = hashlib.md5(file_list_str.encode("utf-8")).hexdigest()
    cache_file = os.path.join(CACHE_FOLDER, f"{hash_val}.pkl")

    # Save the combined dataframe to that file
    combined_df.to_pickle(cache_file)

    # Store or update the DB entry
    _store_cache_entry(file_list_str, cache_file)

    return combined_df

