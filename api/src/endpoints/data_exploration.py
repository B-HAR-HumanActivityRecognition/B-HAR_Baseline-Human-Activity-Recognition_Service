import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from typing import Annotated
from io import BytesIO
import shutil
from pathlib import Path
import zipfile
from modules.data_exploration import pca_plot, sensor_analysis, correlation_matrix, feature_distribution, feature_importance
from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import StreamingResponse
from utility_api.utility import check_bhar_exists
from src.auth.auth import get_current_user

router = APIRouter(
    prefix="/bhar/data_exploration",
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/bhar/auth/token")

@router.get("/")
async def data_exploration_api(
    token: Annotated[str, Depends(oauth2_scheme)],
    name: str,
    sampling_frequency: int,
    segment_duration: float,
):
    """
    **:param name:** name of the B-HAR object \\
    **:param sampling_frequency:** sample frequency in Hz \\
    **:param segment_duration:** duration of the segments in seconds \\
    Computes feature distribution, correlation matrix, feature importance, pca plot and sensor analysis for the B-HAR object with the given name. \\
    Returns a zip file containing the exploration analysis for the B-HAR with the given name.
    """
    username = (await get_current_user(token)).username
    bhar = check_bhar_exists(username, name)
    io = BytesIO()
    folder = "storage/" + username + "/" + name + "_data_exploration"
    Path(folder).mkdir(parents=True, exist_ok=True)
    feature_distribution(bhar._dataframe, folder)
    correlation_matrix(bhar._dataframe, folder)
    feature_importance(bhar._dataframe, folder)
    pca_plot(bhar._dataframe ,sampling_frequency=sampling_frequency, segment_duration=segment_duration, save_to=folder)
    sensor_analysis(bhar._dataframe ,sampling_frequency=sampling_frequency, segment_duration=segment_duration, save_to=folder)
    with zipfile.ZipFile(io, mode='w') as archive:
        for file in os.listdir(folder):
            archive.write(os.path.join(folder, file), file)
    shutil.rmtree(folder)
    return StreamingResponse(
        iter([io.getvalue()]),
        media_type="application/zip",
        headers = {"Content-Disposition":"attachment;filename=output.zip"}
    )