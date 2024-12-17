import requests


# # example
# import random
# nodes = [
#     {"id": 1, "status": "active", "progress": random.randint(0, 100)},
#     {"id": 2, "status": "inactive", "progress": random.randint(0, 100)},
#     {"id": 3, "status": "active", "progress": random.randint(0, 100)}
# ]
# global_progress = random.randint(0, 100)
#
# data = {"nodes": nodes, "global_progress": global_progress}


def web_sender(host, port, target, data):
    # url = f'http://{host}:{port}/{target}'
    # print(url)
    # response = requests.post(url, json=data)
    #
    # # 处理响应
    # if response.status_code == 200:
    #     print("Response JSON:", response.json())
    # else:
    #     print("Error:", response.status_code, response.text)
    pass

def web_getter(host, port, target, data):
    url = f'http://{host}:{port}/{target}'
    print(url)
    response = requests.post(url, json=data)

    # 处理响应
    if response.status_code == 200:
        try:
            return response.json()
        except ValueError:
            return response.text
    else:
        print("Error:", response.status_code, response.text)
        return None
