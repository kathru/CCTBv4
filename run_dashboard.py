import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
uvicorn.run("dashboard.app:app", host="0.0.0.0", port=8000, reload=False)
