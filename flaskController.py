from flask import Flask, render_template, jsonify, request
from flask_restful import Resource, Api, reqparse
import base64
import logging
import logging.handlers
import jsonMethod
import os, sys
import json
import config as cf
import tkinter
from tkinter import filedialog, Toplevel
import shutil
import traceback
import time

import subprocess
import Unet



# Flask
app = Flask(__name__)
api = Api(app)

# 로그 생성
logger = logging.getLogger()

# 로그의 출력 기준 설정
logger.setLevel(logging.INFO)

# logs 출력 형식
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# logs 출력
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# log를 파일에 출력
file_handler = logging.handlers.TimedRotatingFileHandler(
  filename='logs/log_flask', when='midnight', interval=1, delay=True, encoding='utf-8'
  )
file_handler.suffix = '%Y%m%d.txt' # 파일명 끝에 붙여줌; ex. logs-20190811
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route('/')
def home():
    return render_template('home.html')

class getDirPath(Resource):
    def post(self):
        type = request.form['type']
        root = tkinter.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        path = filedialog.askdirectory(parent=root, initialdir=cf.getConfig()["PATH"]["initialdir"], title='폴더 선택')
        path = path.replace("/","\\")
        if path != "":
            cf.setConfigPath("initialdir", "\\".join(path.split("\\")[:-1])) # initialdir 설정
            cf.setConfigPath(type, path) # 환경설정 추가
        return jsonify({ "data" : path })

class getFilePath(Resource):
    def post(self):
        type = request.form['type']
        root = tkinter.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        path = filedialog.askopenfilename(parent=root, initialdir=cf.getConfig()["PATH"]["initialdir"],
                                          title='파일 선택')
        path = path.replace("/", "\\")
        if path != "":
            cf.setConfigPath("initialdir", "\\".join(path.split("\\")[:-1]))  # initialdir 설정
            cf.setConfigPath(type, path)  # 환경설정 추가
        return jsonify({"data": path})

class getFileList(Resource):
    def post(self):
        dict = {}
        if os.path.isdir("./src/model"):
            fileList = []
            for file in os.listdir("./src/model"):
                if file.split(".")[-1] == "pth":
                    fileList.append(file)
        else:
            fileList = []
        return jsonify({ "data" : fileList })

class getConfig(Resource):
    def post(self):
        dirList = {}
        dirList['input'] = cf.getConfig()['PATH']['input']
        dirList['output'] = cf.getConfig()['PATH']['output']
        dirList['test'] = cf.getConfig()['PATH']['test']
        return jsonify({ "data" : dirList })

class delModel(Resource):
    def post(self):
        try:
            modelName = request.form['modelName']
            path = "./src/model/"
            os.remove(path + modelName)
            if os.path.isdir("./src/result/" + modelName):
                shutil.rmtree("./src/result/" + modelName + "/")
            returnDict = {}
            returnDict['result'] = "ok"
        except Exception:
            returnDict['result'] = "error"
        return jsonify(returnDict)

class addModel(Resource):
    def post(self):
        try:
            root = tkinter.Tk()
            root.withdraw()
            root.wm_attributes("-topmost", 1)
            file = filedialog.askopenfile(parent=root, initialdir=cf.getConfig()["PATH"]["initialdir"], title='모델 선택')
            returnDict = {}
            if file!=None:
                cf.setConfigPath("initialdir", "\\".join(file.name.split("/")[:-1]))  # initialdir 설정
                if file.name.split(".")[-1] == 'pth':
                    returnDict['result'] = "ok"
                    shutil.copy2(file.name, "./src/model/" + file.name.split("/")[-1])
                else:
                    returnDict['result'] = "fail"
                    returnDict['message'] = "파일 확장자를 확인해주세요."
            else:
                returnDict['result'] = "empty"
        except Exception:
            returnDict['result'] = "error"
            returnDict['message'] = "시스템 오류가 발생했습니다."
        return jsonify(returnDict)

class getModelImg(Resource):
    def post(self):
        try:
            returnDict = {}
            path = "./src/result/"
            modelName = request.form['modelName']
            if modelName.split(".")[-1] == 'pth': # 파일 확장자명이
                if os.path.isdir(path + modelName):
                    returnDict['MIC1Img'] = os.listdir(path + modelName + "/img/MIC1")
                    returnDict['MIC4Img'] = os.listdir(path + modelName + "/img/MIC4")
                    returnDict['MIC5Img'] = os.listdir(path + modelName + "/img/MIC5")
                    returnDict['result'] = "ok"
                else:
                    returnDict['result'] = "fail"
                    returnDict['message'] = "TEST를 진행하지 않은 모델입니다."
            else:
                returnDict['result'] = "fail"
                returnDict['message'] = "파일 확장자를 확인해주세요."
        except Exception as e:
            logging.info(traceback.format_exc())
            returnDict['result'] = "error"
            returnDict['message'] = "시스템 오류가 발생했습니다."
        return jsonify(returnDict)

class getImgBase64(Resource):
    def post(self):
        filepath = request.get_data().decode('utf-8')
        with open(filepath, 'rb') as img:
            base64_string = base64.b64encode(img.read())

        return jsonify({ "src" : str(base64_string.decode('utf-8')) })

class getProcessingState(Resource):
    def post(self):
        train_data = {}
        test_data = {}
        train_data["TRAIN"] = jsonMethod.getData(type="TRAIN")
        test_data["TEST"] = jsonMethod.getData(type="TEST")
        train_data.update(test_data)
        return jsonify({ "data" : train_data })

class getModelResult(Resource):
    def post(self):
        result = {}
        modelName = request.form['modelName']

        # TEST 결과 조회
        path = "./src/result/" + modelName + "/result.json"
        if os.path.isfile(path):
            with open(path, 'r', encoding="UTF-8") as f:
                json_data = json.load(f)
            result["test_time"] = json_data["time"]
            result["test_num"] = json_data["num"]
            result["test_acc"] = json_data["acc"]
            result["test_loss"] = json_data["loss"]

        # TRAUN 결과 조회
        path = "./src/model/result.json"
        if os.path.isfile(path):
            with open(path, 'r', encoding="UTF-8") as f:
                json_data = json.load(f)
            result["train_time"] = json_data[modelName]["time"]
            result["train_num"] = json_data[modelName]["num"]
            result["train_acc"] = json_data[modelName]["acc"]
            result["train_loss"] = json_data[modelName]["loss"]
        return jsonify({ "data" : result })

class startTrain(Resource):
    def post(self):
        returnDict = {}
        inputPath = request.form['inputPath']
        outputPath = request.form['outputPath']
        ckptName = request.form['ckptName']
        batchSize = request.form['batchSize']
        epochSize = request.form['epochSize']
        if os.path.isfile("./src/model/" + ckptName):
            returnDict["result"] = "fail"
            returnDict["message"] = "모델명이 중복됩니다."
        else:
            Unet.TRAIN(Model_name=ckptName,  # 폴더 생성 이름
                      num_class=4,  # 클래스
                      # trainData_path=outputPath,  # 데이터셋 저장 경로, train, valid 데이터 찾아가서 dataloader 만드는 형식
                      Batch_size=8,  # 배치사이즈
                      Epochs=int(epochSize),  # 에포크
                      LR=0.00068,  # 러닝레이트
                      Deep_supervision=False)

            dict = {}
            dict["view"] = "false"
            dict["total"] = "1"
            dict["count"] = "0"
            jsonMethod.setData(type="TRAIN", data=dict) # TRAIN processingState 초기화
            returnDict['ckptName'] = ckptName + "_model_epoch" + str(epochSize) + ".pth"
            returnDict["result"] = "ok"
            returnDict["message"] = ""
        return jsonify(returnDict)

class startTest(Resource):
    def post(self):
        returnDict = {}
        testPath = request.form['testPath']
        print("testPath : " + testPath)

        ckptName = request.form['ckptName']
        print("ckptName : " + ckptName)
        result_dir = 'D:\\AI_test\\Output2'

        if os.path.isfile("./src/model/" + ckptName):
            if os.path.isdir("./src/result/" + ckptName):
                shutil.rmtree("./src/result/" + ckptName + "/") # 이전에 TEST했던 파일은 삭제처리
                os.mkdir("./src/result/" + ckptName + "")  # 결과이미지 기본 폴더 생성
                os.mkdir("./src/result/" + ckptName + "/img")  # 결과이미지 기본 폴더 생성
                os.mkdir("./src/result/" + ckptName + "/img/MIC1")  # 결과이미지 기본 폴더 생성
                os.mkdir("./src/result/" + ckptName + "/img/MIC4")  # 결과이미지 기본 폴더 생성
                os.mkdir("./src/result/" + ckptName + "/img/MIC5")  # 결과이미지 기본 폴더 생성
            else:
                os.mkdir("./src/result/" + ckptName + "")  # 결과이미지 기본 폴더 생성
                os.mkdir("./src/result/" + ckptName + "/img")  # 결과이미지 기본 폴더 생성
                os.mkdir("./src/result/" + ckptName + "/img/MIC1")  # 결과이미지 기본 폴더 생성
                os.mkdir("./src/result/" + ckptName + "/img/MIC4")  # 결과이미지 기본 폴더 생성
                os.mkdir("./src/result/" + ckptName + "/img/MIC5")  # 결과이미지 기본 폴더 생성

            Unet.Test(Model_name = ckptName)
            # result = "ok"
            # message = "ok"

            dict = {}
            dict["view"] = "false"
            dict["total"] = "1"
            dict["count"] = "0"
            jsonMethod.setData(type="TEST", data=dict)  # TRAIN processingState 초기화
            returnDict["result"] = "ok"
            returnDict["message"] = ""
        else:
            returnDict["result"] = "fail"
            returnDict["message"] = "해당 모델이 존재하지 않습니다. 확인 후 다시 시도해주세요."
        return jsonify(returnDict)

api.add_resource(getDirPath, '/getDirPath') # 폴더경로 조회
api.add_resource(getFilePath, '/getFilePath') # 파일경로 조회
api.add_resource(getFileList, '/getFileList') # 모델 목록조회
api.add_resource(getConfig, '/getConfig') # 환경설정 불러오기
api.add_resource(delModel, '/delModel') # 모델 삭제
api.add_resource(addModel, '/addModel') # 모델 추가
api.add_resource(getModelImg, '/getModelImg') # 이미지 파일 조회
api.add_resource(getImgBase64, '/getImgBase64') # 이미지 base64 변환
api.add_resource(getProcessingState, '/getProcessingState') # 처리현황 조회
api.add_resource(getModelResult, '/getModelResult') # 모델 결과 조회
api.add_resource(startTrain, '/startTrain') # TRAIN 실행
api.add_resource(startTest, '/startTest') # TEST 실행

def flask_start():
    app.run(host="127.0.0.1", port="5001", debug=False)

# flask_start()