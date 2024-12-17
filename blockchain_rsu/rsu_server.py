import json
import argparse
import concurrent.futures
import os
import sys
import threading
import time

import flask
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from block_chain.blockchain import Blockchain
from block_chain.block import Block

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)
from myUtils.RSUMysqlUtil import RSUMysqlUtil
import myUtils.config as mysql_cfg


class RSUServer:
    def __init__(self, name, rsu_server_address, rsu_server_port, cloud_server_address, cloud_server_port, stations):
        self.app = Flask(import_name=name)
        CORS(self.app, supports_credentials=True)
        self.app.config['CORS_HEADERS'] = 'Content-Type'
        self.app.config['CORS_SUPPORTS_CREDENTIALS'] = True
        self.app.config['JSON_SORT_KEYS'] = False
        self.app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

        self.rsu_server_address = rsu_server_address
        self.rsu_server_port = rsu_server_port
        self.cloud_server_address = cloud_server_address
        self.cloud_server_port = cloud_server_port
        self.stations = stations

        self.mysql_util = RSUMysqlUtil(pool_name=f"cloud_server_{cloud_server_port}",
                                       mysql_config=mysql_cfg.mysql_config)
        self.blockchain = Blockchain(self.mysql_util.init_blockchain())

        # Define routes
        @self.app.route('/')
        def index():
            return self.__index()

        @self.app.route('/add/block', methods=['POST'])
        def add_block():
            return self.__add_block()

        @self.app.route('/station', methods=['POST'])
        def station():
            return self.__station()

        @self.app.route('/blockchain/length', methods=['GET'])
        def blockchain_length():
            return self.__route_blockchain_length()

        @self.app.route('/blockchain', methods=['GET'])
        def blockchain():
            return self.__route_blockchain()

        @self.app.route('/blockchain/range', methods=['GET'])
        def blockchain_range():
            return self.__route_blockchain_range()

        @self.app.route('/last/block', methods=['GET'])
        def last_block():
            return self.__last_block()

    def __index(self):
        return 'blockchain is up'

    def __last_block(self):
        return self.blockchain.chain[-1].to_json()

    @staticmethod
    def make_req(url, block_json):
        try:
            result = requests.post(url, json=json.loads(block_json))
            result.raise_for_status()
            return result.json()
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
        except requests.exceptions.JSONDecodeError as e:
            print(f"JSON decode failed: {e}")
            return None

    def __add_block(self):
        data = request.get_json()
        json_string = json.dumps(data, ensure_ascii=False, indent=4)
        block = self.blockchain.add_block(json_string)
        self.mysql_util.add_block_to_mysql(block)

        # Make request to cloud server (non-blocking)
        cloud_url = f'http://{self.cloud_server_address}:{self.cloud_server_port}/add/block'
        threading.Thread(target=self.make_req, args=(cloud_url, json.dumps({'key': block.hash, 'value': data}))).start()

        # Broadcast to other stations (non-blocking)
        urls = [f'http://{self.rsu_server_address}:{port}/station' for port in self.stations if
                port != self.rsu_server_port]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for url in urls:
                executor.submit(self.make_req, url, json.dumps(block.to_json()))

        return flask.jsonify({'status': 'success'})

    def __route_blockchain_length(self):
        return jsonify(len(self.blockchain.chain))

    def __route_blockchain(self):
        return jsonify(self.blockchain.to_json())

    def __route_blockchain_range(self):
        start = int(request.args.get('start'))
        end = int(request.args.get('end'))
        return jsonify(self.blockchain.to_json()[start:end])

    def __station(self):
        data = request.get_json()
        block = Block.from_json(data)
        self.blockchain.add_block_from_block(block)
        return flask.jsonify({"status": 'success'})

    def run(self):
        print('Running server...')
        print(f'Host: {self.rsu_server_address}')
        print(f'Port: {self.rsu_server_port}')
        self.app.run(host=self.rsu_server_address, port=self.rsu_server_port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RSUServer Configuration")
    parser.add_argument('--rsu_server_address', type=str, required=True, help='RSU Server address')
    parser.add_argument('--rsu_server_port', type=int, required=True, help='RSU Server port')
    parser.add_argument('--cloud_server_address', type=str, required=True, help='Cloud Server address')
    parser.add_argument('--cloud_server_port', type=int, required=True, help='Cloud Server port')
    parser.add_argument('--stations', nargs='+', required=True, help='List of station ports')

    args = parser.parse_args()

    server = RSUServer(
        __name__,
        rsu_server_address=args.rsu_server_address,
        rsu_server_port=args.rsu_server_port,
        cloud_server_address=args.cloud_server_address,
        cloud_server_port=args.cloud_server_port,
        stations=args.stations
    )
    server.run()
