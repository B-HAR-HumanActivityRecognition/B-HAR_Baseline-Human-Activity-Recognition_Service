import sys
import os
from typing import Annotated
from fastapi.security import OAuth2PasswordBearer
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from bhar import BHAR
from fastapi import File, UploadFile, APIRouter
from src.models.bhar import BHARModel
import shutil
import pickle
from pathlib import Path
from utility_api.utility import bhars, check_bhar_exists
from fastapi import HTTPException, status, Depends, Body
from src.auth.auth import get_current_user
import kaggle
import time
import pandas as pd

router = APIRouter(
    prefix="/bhar/input",
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/bhar/auth/token")

@router.post("/upload_file", status_code=201)
async def upload(token: Annotated[str, Depends(oauth2_scheme)], file: UploadFile = File(...)):
    """
    **:param Request body:** the file to upload \\
    Uploads a file to the server.
    """
    username = (await get_current_user(token)).username
    try:
        Path("storage/" + username).mkdir(parents=True, exist_ok=True)
        with open("storage/" + username + "/" + file.filename, 'wb') as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()
        
    return {"message": f"Successfully uploaded {file.filename}"}

@router.post("/upload_from_kaggle", status_code=201)
async def upload_kaggle(token: Annotated[str, Depends(oauth2_scheme)], kaggle_file: str):
    """
    **:param kaggle_file:** the part of the url after "kaggle.com/datasets/" \\
    Uploads a dataset from kaggle to the server, unzips it and converts it to a csv file. The files that make up the dataset must be .txt or .csv. \\
    e.g.: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones -> kaggle_file = "uciml/human-activity-recognition-with-smartphones"
    """
    username = (await get_current_user(token)).username
    try:
        Path("storage/" + username).mkdir(parents=True, exist_ok=True)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(kaggle_file, path="storage/" + username + "/kaggle", unzip=True)
        df_tot = pd.DataFrame()
        for dirpath, _, filenames in os.walk("storage/" + username + "/kaggle"):
            for file in filenames:
                if file.endswith(tuple([".csv", ".txt"])):
                    df_tot = pd.concat([df_tot, pd.read_csv(os.path.join(dirpath, file))], ignore_index=True)
        timestamp = time.strftime("%H_%M_%S")
        df_tot.to_csv("storage/" + username + "/kaggle_dataset_" + timestamp + ".csv", index=False)
        shutil.rmtree("storage/" + username + "/kaggle")
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="This dataset is not in the correct format, the files must be .txt or .csv")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
        
    return {"message": f"Successfully uploaded {kaggle_file} with name kaggle_dataset_{timestamp}.csv"}

@router.post("/upload_user_model", status_code=201)
async def upload_user_model(token: Annotated[str, Depends(oauth2_scheme)], file: UploadFile = File(...)):
    """
    **:param Request body:** the model to upload \\
    Uploads a user model to the server. The filename must start with "ml-user_" or "dl-user_" or "dl-hyper-user_". The file must be a pickle (for ml) or h5 (for dl) or py (for dl-hyper) file. \\
    The py file must have one method: def build(hp) -> tf.keras.Model (you can see an example in example/dl-hyper-user_test-model.py on Github)
    """
    username = (await get_current_user(token)).username
    if not file.filename.startswith(tuple(["ml-user_", "dl-user_", "dl-hyper-user_"])):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="The filename must start with ml-user_ or dl-user_ or dl-hyper-user_")
    if not file.filename.endswith(tuple([".pkl", ".h5", ".py"])):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="The file must be a pickle or h5 or py file")
    try:
        Path("storage/" + username).mkdir(parents=True, exist_ok=True)
        with open("storage/" + username + "/" + file.filename, 'wb') as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        print(str(e))
    finally:
        file.file.close()
        
    return {"message": f"Successfully uploaded {file.filename}"}

@router.get("/get_filenames", status_code=200)
async def get_filenames(token: Annotated[str, Depends(oauth2_scheme)]):
    """
    Returns a list of all the names of the files that the user has uploaded.
    """
    username = (await get_current_user(token)).username
    try:
        return [file for file in os.listdir("storage/" + username) if file.endswith(tuple([".csv", ".txt"]))]
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No files for user {username}")

@router.post("/create_bhar", status_code=201)
async def create_bhar(token: Annotated[str, Depends(oauth2_scheme)], bhar_json: BHARModel):
    """
    **:param Request body:** the attributes of the B-HAR object to create as a json \\
    Creates a B-HAR object with the given attributes and saves it in storage as a pickle file.
    """
    username = (await get_current_user(token)).username
    attributes = bhar_json.model_dump()
    try:
        bhar = BHAR(**attributes)
        os.remove(bhar.get_reporter().get_output_file())
        if username != bhar._username:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"User {username} is not allowed to create B-HAR object for user {bhar._username}")
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    
    try:
        for file in os.listdir("storage/" + bhar._username):
            if file.startswith(bhar._name + ".pkl"):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"B-HAR object named {bhar._name} already exists")
    except AttributeError:
        pass
    
    with open(f"storage/{bhar._username}/{bhar._name}.pkl", 'wb') as file:
        pickle.dump(bhar, file)  
    bhars.add(bhar)
    return {"message": f"Created B-HAR: {bhar._name}"}

@router.delete("/delete_bhar", status_code=200)
async def delete_bhar(token: Annotated[str, Depends(oauth2_scheme)], name: str):
    """
    **:param name:** the name of the B-HAR object to delete \\
    Deletes the B-HAR object with the given name.
    """
    username = (await get_current_user(token)).username
    bhar = check_bhar_exists(username, name)
    bhars.remove(bhar)
    os.remove(f"storage/{bhar._username}/{bhar._name}.pkl")
    return {"message": f"Deleted B-HAR: {name}"}

@router.delete("/delete_file", status_code=200)
async def delete_file(token: Annotated[str, Depends(oauth2_scheme)], filename: str):
    """
    **:param filename:** the name of the file to delete \\
    Deletes the file with the given filename.
    """
    username = (await get_current_user(token)).username
    try:
        os.remove(f"storage/{username}/{filename}")
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No file named {filename} for user {username}")
    return {"message": f"Deleted file: {filename}"}

@router.delete("/delete_user_model", status_code=200)
async def delete_user_model(token: Annotated[str, Depends(oauth2_scheme)], name: str):
    """
    **:param name:** the name of the user model to delete \\
    Deletes the user model with the given name.
    """
    username = (await get_current_user(token)).username
    try:
        os.remove(f"storage/{username}/{name}")
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No user model named {name} for user {username}")
    return {"message": f"Deleted user model: {name}"}