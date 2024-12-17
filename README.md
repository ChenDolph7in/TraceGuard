# TraceGuard

## 环境配置

1. 依赖软件

   1. CARLA_0.9.8：[Release CARLA 0.9.8 (development) · carla-simulator/carla (github.com)](https://github.com/carla-simulator/carla/releases/tag/0.9.8/)。安装参考：[Win10下安装CARLA_windows下如何安装carla编译版本-CSDN博客](https://blog.csdn.net/gloria_iris/article/details/128003465?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-128003465-blog-105510833.235^v43^pc_blog_bottom_relevance_base8&spm=1001.2101.3001.4242.1&utm_relevant_index=3)

   2. MYSQL：提前创建数据库，并在`/myUtils`中的`config.py`配置数据库账户信息

      ```sql
      DROP DATABASE TraceGuard;
      CREATE DATABASE IF NOT EXITS TraceGuard;
      ```

2. python环境：

   ```shell
   conda create -n TraceGuard python=3.7
   conda activate TraceGuard
   
   pip install absl-py astunparse cachetools certifi charset-normalizer colorama cycler flatbuffers fonttools gast google-auth google-auth-oauthlib google-pasta grpcio h5py idna importlib-metadata Keras-Preprocessing kiwisolver Markdown matplotlib natsort numpy oauthlib opt-einsum packaging Pillow protobuf pyasn1 pyasn1-modules pyparsing python-dateutil pytz requests requests-oauthlib rsa seaborn six scikit-learn tensorboard tensorboard-data-server tensorboard-plugin-wit tensorboardX tensorflow-estimator termcolor threadpoolctl tqdm typing-extensions urllib3 Werkzeug wrapt zipp tqdm matplotlib pandas scikit-learn joblib tensorflow requests -i https://pypi.tuna.tsinghua.edu.cn/simple
   
   pip install flask requests flask-Cors mysql-connector-python pymysql -i https://pypi.tuna.tsinghua.edu.cn/simple
   pip install pygame future -i https://pypi.tuna.tsinghua.edu.cn/simple
   
   # gpu (for federate learning)
   pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 -i https://pypi.tuna.tsinghua.edu.cn/simple
   # cpu (for federate learning)
   pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 -i https://pypi.tuna.tsinghua.edu.cn/simple
   # 同态加密 CKKS, 使用https://github.com/OpenMined/TenSEAL
   pip install tenseal
   ```
   



## 文件结构

1. 组件

   1. 分层联邦学习
   2. CARLA汽车模拟
   3. RSU/云服务器模拟程序
   4. 区块链
   5. MYSQL连接工具

2. 文件结构

   ```shell
   E:.
   ├─blockchain_rsu         # 区块链和云服务器和RSU服务区程序
   │  ├─block_chain
   │  └─utils
   ├─carla					 # carla库，复制自CARLA_0.9.8/WindowsNoEditor/PythonAPI/carla
   ├─carla_simulation		 # carla模拟程序
   │  └─_out	 # TraceGuard_simulation_picsave.py，模拟摄像头生成的拍摄图片保存目录，图片经过压缩
   ├─hier_federate_learning # 分层联邦学习程序
   │  ├─data	 # 分层联邦学习数据			
   │  │  ├─AV_DATA
   │  │  ├─cifar
   │  │  └─mnist
   │  ├─logs	# 分层联邦学习训练日志		
   │  └─save	# 分层联邦学习训练结果保存		
   │     └─objects
   └─myUtils	# 工具类，如mysql连接等
   
   ```



## 组件

### 1.分层联邦学习

参考：[github.com/Asrua/Hierarchical-Federated-Learning-PyTorch](https://github.com/Asrua/Hierarchical-Federated-Learning-PyTorch)

```shell
baseline_main.py		# 基本的深度学习，没有联邦学习，只是使用模型进行训练

federated_main.py		# 基础联邦学习，N个参与方，一个中心服务器
federated_main_ckks.py  # 基础联邦学习，加入同态加密CKKS

hier_fed_main.py		# 分层联邦学习，共3层，最底层为参与方，最高层为中心服务器，中间一层为中间服务器
hier_fed_main_ckks.py	# 分层联邦学习，加入同态加密CKKS

create_hierarchy.py		# 分层联邦学习中，创建层次结构
create_hierarchy_ckks.py # 分层联邦学习中，创建层次结构，加入同态加密CKKS
```

运行示例：

```shell
# 在CPU运行基础联邦学习
python federated_main.py --model=nn --dataset=av --iid=1 --epochs=10
python federated_main_ckks.py --model=nn --dataset=av --iid=1 --epochs=10
# 在CPU运行分层联邦学习
python hier_fed_main.py --model=nn --dataset=av --iid=1 --epochs=10 
python hier_fed_main_ckks.py --model=nn --dataset=av --iid=1 --epochs=10
# 在GPU运行基础联邦学习
python federated_main.py --model=nn --dataset=av --iid=1 --epochs=10 --gpu=0
python federated_main_ckks.py --model=nn --dataset=av --iid=1 --epochs=10 --gpu=0
# 在GPU运行分层联邦学习
python hier_fed_main.py --model=nn --dataset=av --iid=1 --epochs=10 --gpu=0
python hier_fed_main_ckks.py --model=nn --dataset=av --iid=1 --epochs=10 --gpu=0
```

作品新增内容：

1. 加入使用汽车数据训练集进行训练
2. 加入同态加密CKKS
3. 加入向前端发送数据

### 2.CARLA汽车模拟

```shell
TraceGuard_simulation.py         # 汽车模拟
TraceGuard_simulation_picsave.py # 汽车模拟+摄像头拍摄图片保存
```

新增内容：

1. 加入限制车辆品牌、类型，固定汽车初始位置
2. 加入提取模拟摄像头图片并进行模糊简化，保存到数据集及本地
3. 加入车辆数据提取，并随时保存到数据库，方便前端读取
4. 加入车辆碰撞后信息发送到RSU，并防止发送数据频率过高限制1s内发送一次

### 3.RSU/云服务器-区块链模拟

```shell
cloud_server.py # 云服务器模拟程序
rsu_server.py   # RSU模拟程序
```

### 4.数据库

数据库**TraceGuard**，表结构见每一个组件对应的MysqlUtil内部。共有三个表：

```shell
vehicle_data    # 汽车相关数据，如汽车品牌、速度、加速度、温度、天气等，用于汽车自己缓存和传输给前端展示页面
blockchain		# 区块链数据
vehicle_data _snapshot # 云服务器保存数据，保存发生碰撞时的汽车数据，即vehicle_data数据+对应的key
```



## 整体流程

分层联邦学习和车联网模拟为两个独立过程。

分层联邦学习hier_fed_main流程：
1. 重要参数：
   1. `num_users`：最底层参与方数量
   2. `mid_server`：中间层服务器数量，如[4，2]表示共有三层，最底层为`num_users`，中间层数量为4，最高层数量为2，其中最底层的参与方在参数`num_users`定义数量。
2. 训练过程：
   1. `create_hierarchy`创建层次结构，给每个上层中间服务器分配对应的底层参与方或下层中间服务器
   2. 随机选择N个底层参与方进行训练，训练时下载模型，使用参与方本地数据进行训练
   3. N个参与方在训练完毕后，进行向上递归传递模型参数过程。即在分别进行模型参数聚合后，最底层向中间层传递，中间层向最高层传递

车联网流程：

1. 使用carla汽车模拟，在运行时，将数据传送保存到数据库，前端读取数据库显示到前端
2. 汽车发生碰撞事件时，读取数据库最后一行，将行传输给RSU
3. RSU收到信息后，首先得到存储信息的哈希值作为Key，并在区块链中创建区块保存key；然后将数据传输给中心服务器，中心服务器使用Mysql将key和数据存储在一起；最后将该区块中的数据同步给其他RSU。即区块链保存key，云服务器保存数据。

车联网模拟过程（所有的IP和端口配置在对应目录下`config.py`中，如果在参数中更改port，则在对应config文件中也需要更改）

1. 不保存图片：

   ```shell
   cd blockchain_rsu
   # 开启服务器
   python cloud_server.py --cloud_server_address localhost --cloud_server_port 63211
   # 开启RSU，自动在Mysql中创建table blockchain
   python rsu_server.py --rsu_server_address localhost --rsu_server_port 34567 --cloud_server_address localhost --cloud_server_port 63211 --stations 34567
   # 向blockchain插入第一个初始块，例如（若有数据则不需要这步）
   INSERT INTO blockchain(last_hash, hash) VALUES("1111111111111111","1111111111111112");
   
   cd carla_simulation
   # 启动carla模拟程序
   python TraceGuard_simulation.py 
   ```
   
2. 保存图片：

   ```shell
   cd blockchain_rsu
   # 开启服务器
   python cloud_server.py --cloud_server_address localhost --cloud_server_port 63211
   # 开启RSU，自动在Mysql中创建table blockchain
   python rsu_server.py --rsu_server_address localhost --rsu_server_port 34567 --cloud_server_address localhost --cloud_server_port 63211 --stations 34567
   CTRL + C
   # 向blockchain插入第一个初始块，例如（若有数据则不需要这步）
   INSERT INTO blockchain(last_hash, hash) VALUES("1111111111111111","1111111111111112");
   
   cd carla_simulation
   # 启动carla模拟程序
   python TraceGuard_simulation_picsave.py 
   ```

图片保存格式为base64编码的字符串，从base64编码的字符串转为图片的方法是（假设base64_str为base64编码字符串）：

```python
import base64
import io
from PIL import Image

decoded_data = base64.b64decode(base64_str)
byte_stream = io.BytesIO(decoded_data)
# restored_image = Image.open(byte_stream)
# restored_image.show()
# restored_image.save(byte_stream, format="JPEG", quality=50)
with open(f'_out/current.jpg', 'wb') as f:
    f.write(byte_stream.getvalue())
```

