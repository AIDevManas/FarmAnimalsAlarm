import os


def test_core_files_exist():
    root = os.path.abspath(os.path.dirname(__file__) + os.path.sep + "..")
    assert os.path.exists(os.path.join(root, "app")), "app/ folder should exist"
    assert os.path.exists(os.path.join(root, "requirements.txt")), (
        "requirements.txt should exist"
    )
    # README
    assert os.path.exists(os.path.join(root, "README.md")), "README.md should exist"
