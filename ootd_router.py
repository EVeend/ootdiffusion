from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Response
from typing import Literal

from ootd_service import predict_service

router = APIRouter()

@router.post("/predict")
async def predict(model_type: Literal["hd","dc"],
                mode_image: UploadFile,
                cloth_image: UploadFile,
                category: Literal[0,1,2],
                seed: int):
    
    return predict_service()