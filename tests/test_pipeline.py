import os

def test_data_directory_exists():
    """
    Verify that the primary data directory exists.
    This ensures that data ingestion has a valid destination.
    """
    assert os.path.exists("data"), "[ERROR] The 'data' directory is missing."

def test_src_directory_exists():
    """
    Verify that the source code directory exists.
    """
    assert os.path.exists("src"), "[ERROR] The 'src' directory is missing."

def test_essential_scripts_exist():
    """
    Verify that all critical MLOps scripts are present in the source directory.
    """
    required_scripts = ["ingest_data.py", "preprocess.py", "train.py"]
    
    for script in required_scripts:
        script_path = os.path.join("src", script)
        assert os.path.exists(script_path), f"[ERROR] Required script missing: {script_path}"