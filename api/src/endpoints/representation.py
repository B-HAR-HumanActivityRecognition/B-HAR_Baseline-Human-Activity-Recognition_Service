import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from modules.representation import dataset_representation
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
import pickle
from utility_api.utility import check_bhar_exists, df_as_file, generate_filename
from fastapi.security import OAuth2PasswordBearer
from src.auth.auth import get_current_user
from typing import Annotated

router = APIRouter(
    prefix="/bhar/dataset_representation",
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/bhar/auth/token")

@router.get("", status_code=200)
async def dataset_representation_api(
        bg_tasks: BackgroundTasks,
        token: Annotated[str, Depends(oauth2_scheme)],
        name: str,
        sampling_frequency: int,
        segment_duration: float,
        rep_type: str = 'segmentation',
        segment_duration_overlap: int = 0,
        create_transitions: bool = False
):
    """
    **:param name:** name of the B-HAR object \\
    **:param sampling_frequency:** sample frequency in Hz \\
    **:param segment_duration:** duration of the segments in seconds \\
    **:param rep_type:** type of representation to be applied, it can be {segmentation, features, raw} \\
    Returns the dataset representation of the B-HAR object with the given name as a file.
    """
    username = (await get_current_user(token)).username
    bhar = check_bhar_exists(username, name)
    try:
        bhar._data, bhar._info = dataset_representation(bhar._dataframe, rep_type, sampling_frequency, segment_duration,
                                        segment_duration_overlap, create_transitions)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                        detail=str(e) + f" for B-HAR object named {name}. I suggest you to start with a new B-HAR object and do remove_data_errors first")
    with open(f"storage/{bhar._username}/{bhar._name}.pkl", 'wb') as file:
        pickle.dump(bhar, file)
    path = generate_filename(bhar._username, bhar._name, "Dataset_representation", bhar._dataset_path)
    return df_as_file(bg_tasks, path, bhar._data)