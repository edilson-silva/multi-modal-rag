from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI

load_dotenv(find_dotenv())

api = FastAPI()


@api.get('/')
async def index():
    return {'message': 'API running'}
