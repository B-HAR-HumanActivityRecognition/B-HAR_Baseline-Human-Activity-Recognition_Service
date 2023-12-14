import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import pandas as pd
import zipfile
from fastapi import BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

# A set of all the Bhar objects that have been created, it gets populated when the server starts up
bhars = set()

def generate_filename(username, name, api, dataset_path):
    """
    :param username: username of the user that called the API
    :param name: name of the B-HAR object
    :param api: name of the API that called this method
    :param dataset_path: path of the dataset that was used to create the B-HAR object
    Generates a filename for the given B-HAR object related to the API that called this method.
    """
    return f"{username}_{name}_{api}_{os.path.basename(dataset_path)}"

def df_as_file(bg_tasks: BackgroundTasks, path, df_list):
    """
    :param path: path of the file to be returned
    :param df_list: list of dataframes to be concatenated and saved to the given path
    Concatenates the list of dataframes and saves it to the given path. The file is deleted after the response is sent.
    """
    try:
        df = pd.concat(df_list, ignore_index=True)
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))
    df.to_csv(path, index=False)
    del df
    bg_tasks.add_task(os.remove, path)
    return FileResponse(path, filename=path, media_type="application/octet-stream")

def df_as_zip(bg_tasks: BackgroundTasks, io, filenames, delete_files=True):
    """
    :param io: io object to be used to create the zip
    :param filenames: list of filenames to be zipped
    :param delete_files: if True, deletes the files after the response is sent
    Zips the list of files and returns a StreamingResponse object. The files are deleted after the response is sent if delete_files is True.
    """
    with zipfile.ZipFile(io, mode='w') as archive:
        for file in filenames:
            archive.write(file)
    if delete_files:
        for file in filenames:
            bg_tasks.add_task(os.remove, file)
    return StreamingResponse(
        iter([io.getvalue()]),
        media_type="application/zip",
        headers = {"Content-Disposition":"attachment;filename=output.zip"}
    )

def check_bhar_exists(username, name):
    """
    :param username: username of the user that called the API
    :param name: name of the B-HAR object
    Checks if a B-HAR object with the given name exists for the given user. If it does not, it raises an HTTPException.
    """
    bhar = None
    for bhar_temp in bhars:
        if bhar_temp._name == name and bhar_temp._username == username:
            bhar = bhar_temp
    if bhar is None:
        raise HTTPException(status_code=404, detail=f"B-HAR object named {name} for user {username} not found")
    return bhar