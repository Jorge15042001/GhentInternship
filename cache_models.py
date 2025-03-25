import os
import sqlite3
import pickle
import json
import hashlib

# You can customize these constants as needed
CACHE_FOLDER = "model_cache"
CACHE_DB_PATH = os.path.join(CACHE_FOLDER, "model_cache_db.sqlite")

os.makedirs(CACHE_FOLDER, exist_ok=True)


def _init_cache_db():
    """
    Initialize the SQLite database with two tables:
      1) 'model_cache' for storing trained model paths
      2) 'eval_cache' for storing evaluation results
    """
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()

    # Table for models
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_cache (
        cache_key TEXT PRIMARY KEY,
        pickle_path TEXT
    )
    """)
    
    # Table for evaluation results
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS eval_cache (
        cache_key TEXT PRIMARY KEY,
        pickle_path TEXT
    )
    """)

    conn.commit()
    conn.close()


def _make_hash_key(*items):
    """
    Build a hash key given a set of items. Each item could be:
     - a dictionary of parameters
     - a string (like dataset_id)
     - etc.

    We'll convert them to a canonical JSON string, then hash that string.
    This ensures the same combination produces the same key, regardless of order.

    If you prefer, you can store them in one combined dictionary (model_params, dataset_id, etc.)
    """
    # Convert each item to JSON (sorted keys if dict) so that we get a stable representation
    # Then we join them all with some separator
    items_as_json = []
    for it in items:
        if isinstance(it, dict):
            # stable representation: sort keys
            items_as_json.append(json.dumps(it, sort_keys=True))
        else:
            items_as_json.append(json.dumps(it))  # if it's a str/int/etc.

    combined = "|".join(items_as_json)
    # md5 is quick and easy, but any hash works
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


def _lookup_model_cache(cache_key):
    """Look up a model cache record in the model_cache table by cache_key."""
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT pickle_path FROM model_cache WHERE cache_key = ?", (cache_key,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def _store_model_cache(cache_key, pickle_path):
    """Store or update the record for a model cache entry."""
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO model_cache (cache_key, pickle_path) VALUES (?, ?)
    """, (cache_key, pickle_path))
    conn.commit()
    conn.close()


def _lookup_eval_cache(cache_key):
    """Look up an evaluation cache record in the eval_cache table by cache_key."""
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT pickle_path FROM eval_cache WHERE cache_key = ?", (cache_key,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def _store_eval_cache(cache_key, pickle_path):
    """Store or update the record for an evaluation cache entry."""
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO eval_cache (cache_key, pickle_path) VALUES (?, ?)
    """, (cache_key, pickle_path))
    conn.commit()
    conn.close()


def train_with_cache(model_class, model_params, dataset_id, X_train, y_train=None, force_retrain=False):
    """
    Train a model (subclass of BaseFaultDetectionAlgorithm, or any pickleable model)
    using the specified parameters and dataset_id, with caching.

    - model_class: the actual class (e.g., PCAFaultDetector)
    - model_params: dict of constructor parameters (e.g. {'retained_variance': 0.9, ...})
    - dataset_id: unique string or ID for the training dataset
    - X_train, y_train: training data
    - force_retrain: if True, always retrain and overwrite cache

    Returns: a trained model instance
    """
    _init_cache_db()

    # We'll incorporate the full module+class name to differentiate among classes
    full_class_name = f"{model_class.__module__}.{model_class.__name__}"

    # Build a stable key
    # For example: model class name + model_params + dataset_id
    cache_key = _make_hash_key(full_class_name, model_params, dataset_id)

    # Check if a pickle path already exists in the DB
    existing_path = None if force_retrain else _lookup_model_cache(cache_key)

    if existing_path and os.path.isfile(existing_path):
        # Load from cache
        print(f"[train_with_cache] Loading model from cache: {existing_path}")
        with open(existing_path, "rb") as f:
            model_obj = pickle.load(f)
        return model_obj
    else:
        # Instantiate and train the model
        print("[train_with_cache] Training model from scratch...")
        model_obj = model_class(**model_params)
        model_obj.train(X_train, y_train)

        # Save to pickle
        pickle_path = os.path.join(CACHE_FOLDER, f"model_{cache_key}.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(model_obj, f)

        # Update the DB record
        _store_model_cache(cache_key, pickle_path)

        return model_obj


def load_model_with_cache(model_class, model_params, dataset_id):
    """
    Load a model from cache if it exists. If it does not exist, returns None.
    - model_class
    - model_params
    - dataset_id
    """
    _init_cache_db()
    full_class_name = f"{model_class.__module__}.{model_class.__name__}"
    cache_key = _make_hash_key(full_class_name, model_params, dataset_id)
    existing_path = _lookup_model_cache(cache_key)
    if existing_path and os.path.isfile(existing_path):
        print(f"[load_model_with_cache] Found model: {existing_path}")
        with open(existing_path, "rb") as f:
            model_obj = pickle.load(f)
        return model_obj
    else:
        print("[load_model_with_cache] No cached model found.")
        return None


def evaluate_with_cache(model_obj, dataset_id, eval_params, X_test, y_test, fault_numbers,
                        force_recompute=False):
    """
    Evaluate a trained model, caching the evaluation results (which must be pickleable).
    
    - model_obj: an already-trained model instance
    - dataset_id: unique ID for the test dataset
    - eval_params: dict specifying evaluation parameters (e.g. {"roc_curve": True, ...})
    - X_test, y_test, fault_numbers: test data, labels, fault IDs
    - force_recompute: if True, ignores any cached result and re-runs the evaluation

    Returns: the evaluation result (dict), e.g. from model_obj.evaluate(...)
    """
    _init_cache_db()

    # We'll incorporate the "signature" of the model as well (so if you switch to a different trained model,
    # you won't reuse the old evaluation results).
    # The best approach is to identify the model by the same key used at training time,
    # plus add the model's internal state if needed. For now, let's do a simpler approach:
    # * model_obj can have an attribute ._train_cache_key if you want; or we can do a direct pickle-based hash.
    # We'll just store the model in a temporary pickle for hashing.
    # This ensures two different trained instances won't produce collisions.
    temp_model_bytes = pickle.dumps(model_obj)
    model_hash = hashlib.md5(temp_model_bytes).hexdigest()

    # Build the key for the evaluation
    cache_key = _make_hash_key(model_hash, dataset_id, eval_params)

    # Check DB
    existing_path = None if force_recompute else _lookup_eval_cache(cache_key)

    if existing_path and os.path.isfile(existing_path):
        # Load from cache
        print(f"[evaluate_with_cache] Loading evaluation results from {existing_path}")
        with open(existing_path, "rb") as f:
            results = pickle.load(f)
        return results
    else:
        # Compute the evaluation
        print("[evaluate_with_cache] Computing evaluation results...")
        # You may have different ways of evaluating:
        # for example, if your model has a standard evaluate() method:
        results = model_obj.evaluate(X_test, y_test, fault_numbers, 
                                     roc_curve=eval_params.get('roc_curve', False),
                                     by_fault_type=eval_params.get('by_fault_type', True))

        # Save results
        pickle_path = os.path.join(CACHE_FOLDER, f"eval_{cache_key}.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(results, f)

        _store_eval_cache(cache_key, pickle_path)
        return results
