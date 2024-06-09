import os
from pathlib import Path
import logging

#creating a logging string for log the information
logging.basicConfig(level=logging.INFO,format='[%(asctime)s]:%(message)s:')

#creating a list of file
list_of_files=[
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "research/trai_with_chromaDM.ipynb",
    "app.py",
    "store_index.py",
    "static/.gitkeep",
    "templates/chat.html",
    "test.py"
]

# now we need to create a logic for cretaing these folders structures
for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)
    
    if filedir !="":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"creating directory;{filedir} for the folder;{filename}")
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"creating empty file:{filepath}")
    else:
        logging.info(f"this file {filename} is already created")



