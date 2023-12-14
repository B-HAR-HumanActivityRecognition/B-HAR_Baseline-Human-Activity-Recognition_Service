import sys
import os
from typing import Annotated
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from routes.api import router as api_router
from utility_api.utility import bhars, check_bhar_exists
from src.auth.auth import get_current_user
import pickle

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/bhar/auth/token")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

app.include_router(api_router)

@app.get("/bhar/get_baseline", status_code=200)
async def get_baseline_api(token: Annotated[str, Depends(oauth2_scheme)], name: str):
    """
    **:param name:** name of the B-HAR object \\
    Returns the baseline of the B-HAR object with the given name.
    """
    username = (await get_current_user(token)).username
    bhar = check_bhar_exists(username, name)
    bhar.get_baseline()
    return FileResponse(bhar._reporter.get_output_file(), media_type='application/octet-stream', filename=bhar._reporter.get_output_file(), status_code=201)

@app.get("/bhar/get_bhar_names", status_code=200)
async def get_names(token: Annotated[str, Depends(oauth2_scheme)]):
    """
    Returns a list of all the names of the B-HAR objects that the user has created.
    """
    username = (await get_current_user(token)).username
    return [bhar._username + ": " + bhar._name for bhar in bhars if bhar._username == username]

if __name__ == "__main__":
    try:
        for subdir in os.listdir("storage"):
            for file in os.listdir("storage/" + subdir):
                if file.endswith(".pkl") and not file.startswith("ml-user") and not file.startswith("dl-user") and not file.startswith("dl-user-hyper"):
                    with open("storage/" + subdir + "/" + file, 'rb') as pkl_file:
                        bhar = pickle.load(pkl_file)
                        bhars.add(bhar)
    except Exception as e:
        print(str(e))
    # You can change the IP and port here
    uvicorn.run("main:app")