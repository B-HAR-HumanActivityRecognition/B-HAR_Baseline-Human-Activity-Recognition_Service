import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from fastapi.security import OAuth2PasswordBearer
from src.auth.auth import get_current_user
from typing import Annotated
from io import BytesIO
from modules.cleaning import remove_data_errors, remove_noise
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
import pickle
from utility_api.utility import check_bhar_exists, df_as_file, df_as_zip, generate_filename

router = APIRouter(
    prefix="/bhar/cleaning",
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/bhar/auth/token")

@router.get("/remove_data_errors", status_code=200)
async def remove_data_errors_api(
        bg_tasks: BackgroundTasks,
        token: Annotated[str, Depends(oauth2_scheme)],
        name: str,
        error_sub_method: str = 'mean'
):
    """
    **:param name:** name of the B-HAR object \\
    **:param error_sub_method:** the technique to apply in order to clean the dataset \\
    Removes NaNs or infinite values on the dataset \\
    Returns the cleaned dataframe
    """
    username = (await get_current_user(token)).username
    bhar = check_bhar_exists(username, name)
    bhar._dataframe, _ = remove_data_errors(bhar._dataframe, error_sub_method)
    with open(f"storage/{bhar._username}/{bhar._name}.pkl", 'wb') as file:
        pickle.dump(bhar, file)
    path = generate_filename(bhar._username, bhar._name, "Remove_data_errors", bhar._dataset_path)
    return df_as_file(bg_tasks, path, bhar._dataframe)

@router.get("/remove_noise", status_code=200)
async def remove_noise_api(
        bg_tasks: BackgroundTasks,
        token: Annotated[str, Depends(oauth2_scheme)],
        name: str,
        sample_rate: int,
        filter_name: str,
        cutoff: int or tuple,
        order: int = 4
):
    """
    **:param name:** name of the B-HAR object \\
    **:param sample_rate:** sample frequency of the recorded dataset in Hz \\
    **:param filter_name:** name of the filter to be applied, it can be {lowpass, highpass, bandpass} \\
    **:param cutoff:** the cutoff frequency for the filter. \\
    **:param order:** order of the filter. \\
    Applies filtering to remove noisy data \\
    Returns the filtered dataframe
    """
    username = (await get_current_user(token)).username
    bhar = check_bhar_exists(username, name)
    try:
        bhar._dataframe = remove_noise(bhar._dataframe, sample_rate, filter_name, cutoff, order)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                    detail=str(e) + f" for B-HAR object named {name}. I suggest you to do start with a new B-HAR object and do remove_data_errors first")
    with open(f"storage/{bhar._username}/{bhar._name}.pkl", 'wb') as file:
        pickle.dump(bhar, file)
    path = generate_filename(bhar._username, bhar._name, "Remove_noise", bhar._dataset_path)
    return df_as_file(bg_tasks, path, bhar._dataframe)

@router.get("/data_cleaning", status_code=200)
async def data_cleaning(
        bg_tasks: BackgroundTasks,
        token: Annotated[str, Depends(oauth2_scheme)],
        name: str,
        sample_rate: int,
        filter_name: str,
        cutoff: int or tuple,
        order: int = 4,
        error_sub_method: str = 'mean'
):
    """
    Executes remove_data_errors and remove_noise on the B-HAR object with the given name and returns the resulting dataframes as a zip.
    Check the documentation of the two methods for more details.
    """
    username = (await get_current_user(token)).username
    _ = check_bhar_exists(username, name)
    io = BytesIO()
    file_data_errors = await remove_data_errors_api(bg_tasks, token, name, error_sub_method)
    file_noise = await remove_noise_api(bg_tasks, token, name, sample_rate, filter_name, cutoff, order)
    return df_as_zip(bg_tasks, io, [file_data_errors.filename, file_noise.filename], delete_files=False)