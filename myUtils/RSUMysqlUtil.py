from myUtils.MysqlUtil import MysqlUtil
from datetime import datetime
import time


class RSUMysqlUtil(MysqlUtil):
    def __init__(self, pool_name, mysql_config):
        super().__init__(pool_name, mysql_config)
        self.create_blockchain_table()

    def create_blockchain_table(self):
        connection = self.safe_get_connection()
        cursor = connection.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS blockchain (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_hash VARCHAR(255) NOT NULL,
            hash VARCHAR(255) NOT NULL
        )
        """
        try:
            cursor.execute(create_table_query)
            connection.commit()
        except Exception as e:
            print(f"Error creating saved_data table: {e}")
        finally:
            cursor.close()
            self.safe_close_connection(connection)

    def init_blockchain(self):
        connection = self.safe_get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT timestamp, last_hash, hash FROM blockchain ORDER BY id")
            result = cursor.fetchall()
            cursor.close()
            result_dicts = [{'timestamp': row[0], 'last_hash': row[1], 'hash': row[2]} for row in result]
            return result_dicts
        except Exception as e:
            print(f"Error initializing blockchain: {e}")
        finally:
            self.safe_close_connection(connection)

    def add_block_to_mysql(self, block):
        connection = self.safe_get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute("INSERT INTO blockchain (timestamp, last_hash, hash) VALUES (%s, %s, %s)",
                           (self.nanoseconds_to_timestamp(block.timestamp), block.last_hash, block.hash))
            connection.commit()
            cursor.close()
        except Exception as e:
            print(f"Error adding block to database: {e}")
        finally:
            self.safe_close_connection(connection)

    def nanoseconds_to_timestamp(self, nanosecond_timestamp: int) -> datetime:
        second_timestamp = nanosecond_timestamp / 1_000_000_000

        return datetime.fromtimestamp(second_timestamp)

    def timestamp_to_nanoseconds(self, timestamp: datetime) -> int:
        second_timestamp = timestamp.timestamp()

        return int(second_timestamp * 1_000_000_000)
