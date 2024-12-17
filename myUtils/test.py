import json

import config as cfg
from myUtils.AVMysqlUtil import AVMysqlUtil

db_manager = AVMysqlUtil(pool_name='vehicle', mysql_config=cfg.mysql_config, save_image=True)
datas = db_manager.get_data(last_only=True, with_image=True)
for data in datas:
    data['timestamp'] = data['timestamp'].isoformat()

try:
    json_data = json.dumps(datas, ensure_ascii=False, default=str)  # 使用default=str确保时间戳等格式转换正常
    print("JSON 格式的最后 30 行数据：")
    print(json_data)
except Exception as e:
    print(f"Error converting data to JSON: {e}")