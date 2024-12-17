import json
from myUtils.MysqlUtil import MysqlUtil


class CloudServerMysqlUtil(MysqlUtil):
    def __init__(self, pool_name, mysql_config):
        super().__init__(pool_name, mysql_config)
        self.create_snapshot_table()

    def create_snapshot_table(self):
        connection = self.safe_get_connection()
        cursor = connection.cursor()

        create_table_query = """
        CREATE TABLE IF NOT EXISTS vehicle_data_snapshot (
            id INT AUTO_INCREMENT PRIMARY KEY,
            snapshot_key VARCHAR(100) NOT NULL UNIQUE,  
            snapshot_data JSON,                        
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
        )
        """

        try:
            cursor.execute(create_table_query)
            connection.commit()
        except Exception as e:
            print(f"Error creating vehicle_data_snapshot table: {e}")
        finally:
            cursor.close()
            self.safe_close_connection(connection)

    def add_data_to_mysql(self, snapshot_key, data):
        connection = self.safe_get_connection()
        cursor = connection.cursor()

        try:
            snapshot_data = json.dumps(data, ensure_ascii=False, default=str)

            insert_query = """
            INSERT INTO vehicle_data_snapshot (snapshot_key, snapshot_data)
            VALUES (%s, %s)
            """
            cursor.execute(insert_query, (snapshot_key, snapshot_data))
            connection.commit()
            print(f"Data successfully added to vehicle_data_snapshot with key {snapshot_key}")
        except Exception as e:
            print(f"Error adding JSON data to vehicle_data_snapshot: {e}")
        finally:
            cursor.close()
            self.safe_close_connection(connection)

    def get_data_from_mysql(self, snapshot_key: str):
        connection = self.safe_get_connection()
        cursor = connection.cursor(dictionary=True)
        try:
            select_query = """
            SELECT snapshot_key, snapshot_data 
            FROM vehicle_data_snapshot 
            WHERE snapshot_key = %s
            """
            cursor.execute(select_query, (snapshot_key,))
            result = cursor.fetchone()

            if result:
                snapshot_data = result['snapshot_data']
                try:
                    snapshot_data = json.loads(snapshot_data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON data: {e}")
                    snapshot_data = None
                return {"snapshot_key": result['snapshot_key'],
                        "snapshot_data": snapshot_data}
            else:
                return None
        except Exception as e:
            print(f"Error retrieving data from vehicle_data_snapshot: {e}")
            return None
        finally:
            cursor.close()
            self.safe_close_connection(connection)
