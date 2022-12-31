import torch
import pyannote.audio
# from pyannote.audio import Pipeline
import time
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from pydub import AudioSegment
import os
import shutil

def vadoutput(filepath):
    access_token="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"
    # pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2022.07",use_auth_token=access_token)
    start=time.time()
    model = Model.from_pretrained("pyannote/segmentation", 
                              use_auth_token=access_token)
    # print(model)
    # diarization = pipeline("/content/new.wav")
    # with open("audio.rttm", "w") as rttm:
    #     diarization.write_rttm(rttm)
    end=time.time()
    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {
  # onset/offset activation thresholds
  "onset": 0.5, "offset": 0.5,
  # remove speech regions shorter than that many seconds.
  "min_duration_on": 0.0,
  # fill non-speech regions shorter than that many seconds.
  "min_duration_off": 0.0
  # "instantiated":True
}
    pipeline.instantiate(HYPER_PARAMETERS)
    vad = pipeline(filepath)
    # return vad.for_json()
    return vad.for_json()


app=FastAPI()

@app.post("/")
async def create_vad_timestamps(file: UploadFile = File(...)):
    filepath="D:\saptangapi\.venv"
    filepath=filepath+"\\"
    filepath=filepath+file.filename
    # # save(filepath)

    # wav_file = AudioSegment.from_file(file,
    #                               format = "wav")
    with open(".venv\destination.wav", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    isExist = os.path.exists(".venv\destination.wav")
    if(isExist==True):
         sol= vadoutput(".venv\destination.wav")
         return {"filename": sol["content"]}
    else:
        return {"Error": f"{file.filename} does not exist."}