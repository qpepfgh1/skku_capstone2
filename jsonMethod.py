import json
import redis

pool = redis.ConnectionPool(host='localhost', port=6379, db=0)


def getData(type=""):
    r = redis.Redis(connection_pool=pool)
    data = r.get(type)
    if data == None:
        setData(type=type, data={})
        data = r.get(type)
    data = json.loads(data.decode('utf-8'))
    return data

def setData(type="", data={}):
    r = redis.Redis(connection_pool=pool)
    data = json.dumps(data, ensure_ascii=False).encode('utf-8')
    r.set(type,data)

def emptyData(type=""):
    json_data = getData(type=type)
    json_data['view'] = "false"
    json_data['state'] = "good"
    json_data['title'] = "정상"
    json_data['test.h5'] = ""
    json_data['detail_text'] = "시스템이 정상 작동중입니다."
    setData(type=type, data=json_data)