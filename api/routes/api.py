import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from fastapi import APIRouter
from src.auth import auth
from src.endpoints import input, cleaning, representation, train_and_test, preprocessing, data_exploration

router = APIRouter()
router.include_router(input.router)
router.include_router(cleaning.router)
router.include_router(preprocessing.router)
router.include_router(representation.router)
router.include_router(train_and_test.router)
router.include_router(data_exploration.router)
router.include_router(auth.router)