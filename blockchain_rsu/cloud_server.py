import json
import argparse
import os
import sys

from flask import Flask, jsonify, request

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)
from myUtils.CloudServerMysqlUtil import CloudServerMysqlUtil
import myUtils.config as mysql_cfg


class CloudServer:
    def __init__(self, name, cloud_server_address, cloud_server_port):
        self.app = Flask(import_name=name)
        self.app.config['JSON_SORT_KEYS'] = False
        self.app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
        self.app.config['CORS_HEADERS'] = 'Content-Type'
        self.app.config['CORS_SUPPORTS_CREDENTIALS'] = True

        self.cloud_server_address = cloud_server_address
        self.cloud_server_port = cloud_server_port

        self.mysql_util = CloudServerMysqlUtil(pool_name=f"cloud_server_{cloud_server_port}",
                                               mysql_config=mysql_cfg.mysql_config)

        @self.app.route('/')
        def index():
            return self.__index()

        @self.app.route('/add/block', methods=['POST'])
        def add_block():
            return self.__add_block()

        @self.app.route('/get/block', methods=['POST'])
        def get_block():
            return self.__get_block()

    def __index(self):
        return 'cloud server is up'

    def __add_block(self):
        data = request.get_json()
        key = data['key']
        value = data['value']
        self.mysql_util.add_data_to_mysql(key, value)
        return jsonify({'status': 'success'})

    def __get_block(self):
        key = request.get_json()['key']
        print('receive key:', key)
        result = self.mysql_util.get_data_from_mysql(key)
        return jsonify(result)

    def run(self):
        print('Running server...')
        print(f'Host: {self.cloud_server_address}')
        print(f'Port: {self.cloud_server_port}')
        self.app.run(host=self.cloud_server_address, port=self.cloud_server_port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CloudServer Configuration")
    parser.add_argument('--cloud_server_address', type=str, required=True, help='Cloud Server address')
    parser.add_argument('--cloud_server_port', type=int, required=True, help='Cloud Server port')

    args = parser.parse_args()

    server = CloudServer(__name__, args.cloud_server_address, args.cloud_server_port)
    server.run()
