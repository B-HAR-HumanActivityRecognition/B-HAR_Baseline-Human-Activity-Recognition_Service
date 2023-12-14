import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import pandas as pd
from io import BytesIO
from typing import Annotated
from modules.preprocessing import drop_data, train_test_splitting, normalization, features_selection, balancing
from fastapi import APIRouter, BackgroundTasks, HTTPException, status, Body, Depends
import pickle
from utility_api.utility import df_as_zip, generate_filename, check_bhar_exists
from fastapi.security import OAuth2PasswordBearer
from src.auth.auth import get_current_user

router = APIRouter(
    prefix="/bhar/preprocessing",
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/bhar/auth/token")

@router.post("/drop_data", status_code=200)
async def drop_data_api(
        bg_tasks: BackgroundTasks,
        token: Annotated[str, Depends(oauth2_scheme)],
        name: str,
        to_drop: dict = Body(...)
):
    """
    **IMPORTANT: Before calling this method, you need to call dataset_representation first.** \\
    **:param name:** name of the B-HAR object \\
    **:param Request body:** a dictionary containing the info to be dropped, for example {"label": ["Walking", "Sitting"]} \\
    Drops some information from the dataset of the B-HAR object, these info can be label values, subject or sessions \\
    Returns the data and info dataframes as files inside a zip.
    """
    username = (await get_current_user(token)).username
    bhar = check_bhar_exists(username, name)
    io = BytesIO()
    if bhar._data is None or bhar._info is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"B-HAR object named {name} has no data or info, you need to call dataset_representation first")
    bhar._data, bhar._info = drop_data(bhar._data, bhar._info, to_drop)
    with open(f"storage/{bhar._username}/{bhar._name}.pkl", 'wb') as file:
        pickle.dump(bhar, file)   
    filenames = [generate_filename(bhar._username, bhar._name, "Selected_data", bhar._dataset_path), generate_filename(bhar._username, bhar._name, "Selected_info", bhar._dataset_path)]
    bhar._data.to_csv(filenames[0], index=False)
    bhar._info.to_csv(filenames[1], index=False)
    return df_as_zip(bg_tasks, io, filenames)

@router.post("/train_test_splitting", status_code=200)
async def train_test_splitting_api(
        bg_tasks: BackgroundTasks,
        token: Annotated[str, Depends(oauth2_scheme)],
        name: str,
        method: str = 'normal',
        test_size: float = .2,
        filter_dict: dict = None
):
    """
    **IMPORTANT: Before calling this method, you need to call dataset_representation first.** \\
    **:param name:** name of the B-HAR object \\
    **:param method:** the methodology to accomplish the train test split, can be: {normal, intra or leave-out} \\
    **:param test_size:** used when method is set to 'normal', it represents the percentage of data in the test set \\
    **:param Request body:** used when method is set to 'leave-out', put in train set the following rows {"subject": [194], "label": ["Walking" , "Jogging"]} \\
    Splits the dataset of the B-HAR object into two subset one for training and one for testing \\
    Returns the train and test dataframes as files inside a zip.
    """
    username = (await get_current_user(token)).username
    bhar = check_bhar_exists(username, name)
    io = BytesIO()
    if bhar._data is None or bhar._info is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"B-HAR object named {name} has no data or info, you need to call dataset_representation first")
    bhar._X_train, bhar._X_test, bhar._y_train, bhar._y_test = train_test_splitting(
        bhar._data,
        bhar._info,
        method=method,
        test_size=test_size,
        filter_dict=filter_dict
    )
    with open(f"storage/{bhar._username}/{bhar._name}.pkl", 'wb') as file:
        pickle.dump(bhar, file)   

    filenames = [generate_filename(bhar._username, bhar._name, "X_train", bhar._dataset_path), generate_filename(bhar._username, bhar._name, "X_test", bhar._dataset_path),
                  generate_filename(bhar._username, bhar._name, "y_train", bhar._dataset_path), generate_filename(bhar._username, bhar._name, "y_test", bhar._dataset_path)]
    bhar._X_train.to_csv(filenames[0], index=False)
    bhar._X_test.to_csv(filenames[1], index=False)
    bhar._y_train.to_csv(filenames[2], index=False)
    bhar._y_test.to_csv(filenames[3], index=False)
    return df_as_zip(bg_tasks, io, filenames)

@router.get("/normalization")
async def normalization_api(
        bg_tasks: BackgroundTasks,
        token: Annotated[str, Depends(oauth2_scheme)],
        name: str,
        method: str = 'min-max',
):
    """
    **IMPORTANT: Before calling this method, you need to call dataset_representation first.** \\
    **:param name**: name of the B-HAR object \\
    **:param method:** normalization method chosen {min-max, robust, standard} \\
    **:param representation:** data representation chosen {segmentation, features, raw} \\
    Normalizes the data of the B-HAR object using training set as reference \\
    Returns the normalized train and test dataframes of the Bhar object with the given name as files inside a zip.
    """
    username = (await get_current_user(token)).username
    bhar = check_bhar_exists(username, name)
    io = BytesIO()
    if bhar._X_train is None or bhar._X_test is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"B-HAR object named {name} has no train or test data, you need to call train_test_splitting first")
    try:
        bhar._X_train, bhar._X_test = normalization(
            bhar._X_train,
            bhar._X_test,
            n_data_cols=bhar._data_shape,
            method=method,
            representation=bhar._representation
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                            detail=str(e) + f" for B-HAR object named {name}. I suggest you to start with a new B-HAR object and do remove_data_errors first")

    with open(f"storage/{bhar._username}/{bhar._name}.pkl", 'wb') as file:
        pickle.dump(bhar, file)   

    filenames = [generate_filename(bhar._username, bhar._name, "Norm_train_df", bhar._dataset_path), generate_filename(bhar._username, bhar._name, "Norm_test_df", bhar._dataset_path)]
    bhar._X_train.to_csv(filenames[0], index=False)
    bhar._X_test.to_csv(filenames[1], index=False)
    return df_as_zip(bg_tasks, io, filenames)

@router.get("/features_selection")
async def features_selection_api(
        bg_tasks: BackgroundTasks,
        token: Annotated[str, Depends(oauth2_scheme)],
        name: str,
        method: str,
):
    """
    **IMPORTANT: Before calling this method, you need to call train_test_splitting first.** \\
    **:param name:** name of the B-HAR object \\
    **:param method:** the methodology to accomplish the feature selection, can be: {variance, l1, tree-based, recursive} \\
    Returns the train and test dataframes with the selected features of the B-HAR object with the given name as files inside a zip.
    """
    username = (await get_current_user(token)).username
    bhar = check_bhar_exists(username, name)
    io = BytesIO()
    if bhar._X_train is None or bhar._X_test is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"B-HAR object named {name} has no train or test data, you need to call train_test_splitting first")
    try:
        bhar._X_train, bhar._X_test = features_selection(
            bhar._X_train,
            bhar._X_test,
            method=method,
            train_info=bhar._y_train
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                            detail=str(e) + f" for B-HAR object named {name}. I suggest you to start with a new B-HAR object and do remove_data_errors first")
    
    with open(f"storage/{bhar._username}/{bhar._name}.pkl", 'wb') as file:
        pickle.dump(bhar, file)   
    filenames = [generate_filename(bhar._username, bhar._name, "F_s_train_df", bhar._dataset_path), generate_filename(bhar._username, bhar._name, "F_s_test_df", bhar._dataset_path)]
    if isinstance(bhar._X_train, pd.DataFrame):
        bhar._X_train.to_csv(filenames[0], index=False)
    else:
        bhar._X_train.tofile(filenames[0], sep=",")
    if isinstance(bhar._X_test, pd.DataFrame):
        bhar._X_test.to_csv(filenames[1], index=False)
    else:
        bhar._X_test.tofile(filenames[1], sep=",")
    return df_as_zip(bg_tasks, io, filenames)

@router.get("/balancing")
async def balancing_api(
        bg_tasks: BackgroundTasks,
        token: Annotated[str, Depends(oauth2_scheme)],
        name: str,
        method: str,
):
    """
    **IMPORTANT: Before calling this method, you need to call train_test_splitting first.** \\
    **:param name:** name of the B-HAR object \\
    **:param method:** the methodology to accomplish the balancing, can be: {smote, random_under, near_miss, edited_nn, adasyn, kmeans_smote, random_over} \\
    Returns the balanced train and test dataframes of the B-HAR object with the given name as files inside a zip.
    """
    username = (await get_current_user(token)).username
    bhar = check_bhar_exists(username, name)
    io = BytesIO()
    if bhar._X_train is None or bhar._y_train is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"B-HAR object named {name} has no train data, you need to call train_test_splitting first")
    bhar._y_train = bhar._y_train['label'].to_frame()
    try:
        bhar._X_train, bhar._y_train = balancing(
            bhar._X_train,
            bhar._y_train,
            method=method
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                            detail=str(e) + f" for B-HAR object named {name}. I suggest you to start with a new B-HAR object and do remove_data_errors first")
    with open(f"storage/{bhar._username}/{bhar._name}.pkl", 'wb') as file:
        pickle.dump(bhar, file)    
    filenames = [generate_filename(bhar._username, bhar._name, "Balancing_train_df", bhar._dataset_path), generate_filename(bhar._username, bhar._name, "Balancing_train_info", bhar._dataset_path)]
    if isinstance(bhar._X_train, pd.DataFrame):
        bhar._X_train.to_csv(filenames[0], index=False)
    else:
        bhar._X_train.tofile(filenames[0], sep=",")
    if isinstance(bhar._X_test, pd.DataFrame):
        bhar._X_test.to_csv(filenames[1], index=False)
    else:
        bhar._X_test.tofile(filenames[1], sep=",")
    return df_as_zip(bg_tasks, io, filenames)

@router.post("/data_engineering", status_code=200)
async def data_engineering(
        bg_tasks: BackgroundTasks,
        token: Annotated[str, Depends(oauth2_scheme)],
        name: str,
        features_selection_method: str,
        balancing_method: str,
        normalization_method: str = 'min-max',
        train_test_splitting_method: str = 'normal',
        test_size: float = .2,
        filter_dict: dict = None,
        to_drop: dict = Body(...)
):
    """
    **IMPORTANT: Before calling this method, you need to call dataset_representation and train_test_splitting first.** \\
    Executes all of the above preprocessing methods and returns the results as files inside a zip. \\
    Check the documentation of the above methods for more details.
    """
    username = (await get_current_user(token)).username
    bhar = check_bhar_exists(username, name)
    io = BytesIO()
    if bhar._data is None or bhar._info is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"BHAR object named {name} has no data or info, you need to call dataset_representation first")
    if bhar._X_train is None or bhar._X_test is None or bhar._y_train is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"BHAR object named {name} has no train or test data, you need to call train_test_splitting first")
    await drop_data_api(bg_tasks, token, name, to_drop)
    await train_test_splitting_api(bg_tasks, token, name, train_test_splitting_method, test_size, filter_dict)
    await normalization_api(bg_tasks, token, name, normalization_method)
    await features_selection_api(bg_tasks, token, name, features_selection_method)
    await balancing_api(bg_tasks, token, name, balancing_method)
    filenames = [generate_filename(bhar._username, bhar._name, "Selected_data", bhar._dataset_path), generate_filename(bhar._username, bhar._name, "Selected_info", bhar._dataset_path),
                generate_filename(bhar._username, bhar._name, "X_train", bhar._dataset_path), generate_filename(bhar._username, bhar._name, "X_test", bhar._dataset_path),
                generate_filename(bhar._username, bhar._name, "y_train", bhar._dataset_path), generate_filename(bhar._username, bhar._name, "y_test", bhar._dataset_path),
                generate_filename(bhar._username, bhar._name, "Norm_train_df", bhar._dataset_path), generate_filename(bhar._username, bhar._name, "Norm_test_df", bhar._dataset_path),
                generate_filename(bhar._username, bhar._name, "F_s_train_df", bhar._dataset_path), generate_filename(bhar._username, bhar._name, "F_s_test_df", bhar._dataset_path),
                generate_filename(bhar._username, bhar._name, "Balancing_train_df", bhar._dataset_path), generate_filename(bhar._username, bhar._name, "Balancing_train_info", bhar._dataset_path)]

    return df_as_zip(bg_tasks, io, filenames, delete_files=False)