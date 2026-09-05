"""
SUBTEXT NLP Research Framework
Context-Aware Online Distress Severity Detection
"""

import os
import sys

# Load environment variables if .env exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# On Windows with Python 3.8+, register torch/lib DLL directory to allow C++ runtime DLLs to be loaded
if sys.platform == "win32":
    appdata = os.environ.get("APPDATA", "")
    torch_lib = os.path.join(appdata, "Python", f"Python{sys.version_info.major}{sys.version_info.minor}", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass

__version__ = "0.1.0"
