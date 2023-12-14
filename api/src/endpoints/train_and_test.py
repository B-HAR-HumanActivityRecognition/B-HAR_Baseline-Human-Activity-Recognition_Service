import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from typing import Annotated
from models.machine_learning_hypermodels import DecisionTree, KNN, RandomForest, SupportVectorClassification, WKNN
from models.deep_learning_hypermodels import LightCNNHyperModel, CNNHyperModel, RNNHyperModel, LSTMHyperModel
from models.user_models import DLUserHyperModel, DLUserModel, MLUserModel
from fastapi.security import OAuth2PasswordBearer
import zipfile
import pandas as pd
from keras.models import load_model
import pickle
from fastapi.responses import StreamingResponse
from utility import reporter
from io import BytesIO
from modules.train_and_test import ModelsManager
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from utility_api.utility import check_bhar_exists
from src.auth.auth import get_current_user

router = APIRouter(
    prefix="/bhar/start_train_and_test",
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/bhar/auth/token")

@router.get("", status_code=201)
async def start_train_and_test_api(
    bg_tasks: BackgroundTasks,
    token: Annotated[str, Depends(oauth2_scheme)],
    name: str,
    KNN_bool: bool = False,
    RandomForest_bool: bool = False,
    DecisionTree_bool: bool = False,
    SupportVectorClassification_bool: bool = False,
    WKNN_bool: bool = False,
    # LDA_bool: bool = False, NOT WORKING
    # QDA_bool: bool = False, NOT WORKING
    LightCNNHyperModel_bool: bool = False,
    CNNHyperModel_bool: bool = False,
    RNNHyperModel_bool: bool = False,
    LSTMHyperModel_bool: bool = False,
    user_models_bool: bool = False
):
    """
    **IMPORTANT: Before calling this method, you need to call train_test_splitting first.** \\
    **:param name:** name of the B-HAR object \\
    **:param ???_bool:** if True, ??? will be used for the training and testing process \\
    Starts the training and testing process for the B-HAR object with the given name and for the selected models. \\
    Returns a zip file containing the training and testing results and the models.
    """
    username = (await get_current_user(token)).username
    bhar = check_bhar_exists(username, name)
    io = BytesIO()
    if bhar._X_train is None or bhar._X_test is None or bhar._y_train is None or bhar._y_test is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"B-HAR object named {name} has no train or test data, you need to call train_test_splitting first")
    shape = (0, 0)
    if bhar._representation == 'raw':
        pass
    if bhar._representation == 'segmentation':
        if isinstance(bhar._X_train, pd.DataFrame):
            bhar._X_train = bhar._X_train.values
        else:
            bhar._X_train = bhar._X_train

        if isinstance(bhar._X_test, pd.DataFrame):    
            bhar._X_test = bhar._X_test.values
        else:
            bhar._X_test = bhar._X_test
        shape = bhar._get_shape_cnn()

    num_classes = bhar._get_num_classes(bhar._target_feature)
    new_reporter = reporter.Reporter(ref_file=bhar._dataset_path, bhar_name=bhar._name, bhar_username=bhar._username,
                report_file_name=f"{bhar._username}_{bhar._name}_train_and_test_results.txt", output_directory=os.getcwd())
    os.remove(new_reporter.get_output_file())
    models = {'ml': [], 'dl': [], 'dl-hyper-user': [], 'dl-user': [], 'ml-user': []}
    if KNN_bool:
        models['ml'].append(KNN(bhar._username))
    if RandomForest_bool:
        models['ml'].append(RandomForest(bhar._username))
    if SupportVectorClassification_bool:
        models['ml'].append(SupportVectorClassification(bhar._username))
    if DecisionTree_bool:
        models['ml'].append(DecisionTree(bhar._username))
    if WKNN_bool:
        models['ml'].append(WKNN(bhar._username))
    if LightCNNHyperModel_bool:
        models['dl'].append(LightCNNHyperModel(bhar._username, shape, num_classes, new_reporter.get_dataset_name()))
    if CNNHyperModel_bool:
        models['dl'].append(CNNHyperModel(bhar._username, shape, num_classes, new_reporter.get_dataset_name()))
    if RNNHyperModel_bool:
        models['dl'].append(RNNHyperModel(bhar._username, shape, num_classes, new_reporter.get_dataset_name()))
    if LSTMHyperModel_bool:
        models['dl'].append(LSTMHyperModel(bhar._username, shape, num_classes, new_reporter.get_dataset_name()))
    if user_models_bool:
        for file in os.listdir(f"storage/{bhar._username}"):
            if file.startswith("ml-user_"):
                with open(f"storage/{bhar._username}/{file}", 'rb') as pkl_file:
                    user_model = pickle.load(pkl_file)
                    model_temp = MLUserModel(bhar._username, file.split('_')[0], user_model)
                    models["ml-user"].append(model_temp)
            elif file.startswith("dl-user_"):
                user_model = load_model(f"storage/{bhar._username}/{file}")
                model_temp = DLUserModel(bhar._username, file.partition("_")[-1].rsplit('.',1)[0], 
                                         user_model, shape, num_classes)
                models["dl-user"].append(model_temp)
            elif file.startswith("dl-hyper-user_"):
                #user_model = load_model(f"storage/{bhar._username}/{file}")
                model_temp = DLUserHyperModel(bhar._username, file.partition("_")[-1].rsplit('.',1)[0], file,
                                              shape, num_classes, new_reporter.get_dataset_name())
                models["dl-hyper-user"].append(model_temp)
    if models.values() == [[], [], [], [], []]:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="You need to select at least one model")
    if user_models_bool and models["dl-user"] == [] and models["dl-hyper-user"] == [] and models["ml-user"] == []:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No user model found, you need to upload it first")
    bhar._models_manager = ModelsManager(shape, num_classes, new_reporter, models)

    # useless
    # if bhar._user_models is not None:
    #     for usr_model in bhar._user_models:
    #         bhar._models_manager.add_model(usr_model)

    bhar._models_manager.start_train_and_test(
        x_train=bhar._X_train,
        x_test=bhar._X_test,
        y_train=bhar._y_train[bhar._target_feature].values,
        y_test=bhar._y_test[bhar._target_feature].values,
        representation=bhar._representation
    )
    bhar._models_manager.export_models()
    filenames = [f"{bhar._username}_{bhar._name}_train_and_test_results.txt"]
    for model_list in bhar._models_manager.get_models().values():
        for model in model_list:
            if model in bhar._models_manager.get_models()["ml"] or model in bhar._models_manager.get_models()["ml-user"]:
                filenames.append(f"{bhar._username}_{model.name}.pkl")
            else:
                filenames.append(f"{bhar._username}_{model.name}_weights.h5")
                
    with zipfile.ZipFile(io, mode='w') as archive:
        for file in filenames:
            archive.write(file)

    for file in filenames:
        bg_tasks.add_task(os.remove, file)

    return StreamingResponse(
        iter([io.getvalue()]),
        media_type="application/zip",
        headers = {"Content-Disposition":"attachment;filename=output.zip"}
    )