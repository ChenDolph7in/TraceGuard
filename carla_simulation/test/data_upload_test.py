import json
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)
import carla_simulation.config as cfg

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)
from myUtils.WebSenderUtil import web_sender
from myUtils.AVMysqlUtil import AVMysqlUtil
import myUtils.config as mysql_cfg

db_manager = AVMysqlUtil(pool_name='vehicle', mysql_config=mysql_cfg.mysql_config, save_image=True)
datas = db_manager.get_data(last_only=True, with_image=True)

json_data = ""
for data in datas:
    data['timestamp'] = data['timestamp'].isoformat()

try:
    json_data = json.dumps(datas, ensure_ascii=False, default=str)
except Exception as e:
    print(f"Error converting data to JSON: {e}")

start_time = time.time()

web_sender(host=cfg.RSU_host, port=cfg.RSU_port, target=cfg.target, data=json_data)

end_time = time.time()

elapsed_time = end_time - start_time
print(f"web_sender 调用耗时: {elapsed_time}秒")
