from dotenv import load_dotenv
import os
load_dotenv()

MODEL_NAME_OR_PATH=os.getenv('MODEL_NAME_OR_PATH')
ID_HOST =os.getenv('ID_HOST')
ID_PORT =os.getenv('ID_PORT')