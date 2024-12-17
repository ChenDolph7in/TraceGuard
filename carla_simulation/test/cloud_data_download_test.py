import json
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)
from myUtils.WebSenderUtil import web_getter

data = {}
data['key'] = '7b290e4da451faf18e67ac135d6c76a3de7db07af9dc23ce16f3b8f107aa9975'

start_time = time.time()
response = web_getter(host='127.0.0.1', port=63211, target='get/block', data=data)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"web_getter 调用耗时: {elapsed_time}秒")

# print(data)
# data = response['snapshot_data']
# print(response['snapshot_data'])
# a = json.loads(data)
# print(a)
# print(type(a))
# for i in a:
#     print(i)

