import sys
import os
py_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../codes'))

if py_folder not in sys.path:
    sys.path.insert(0, py_folder)
print(f"Added codes folder to sys.path: {py_folder}")
