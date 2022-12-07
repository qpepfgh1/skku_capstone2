import json

def getConfig():
    with open('./conf/config.json', 'r', encoding="UTF-8") as f:
        json_data = json.load(f)
    data = json_data
    return data

def setConfigPath(type, path):
    data = getConfig()
    data['PATH'][type] = path
    with open('./conf/config.json', 'w', encoding="UTF-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False)