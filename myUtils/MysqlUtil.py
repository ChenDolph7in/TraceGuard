import mysql.connector.pooling


class MysqlUtil:
    def __init__(self, pool_name: str, mysql_config: dict):
        self.pool_name = pool_name
        self.mysql_config = mysql_config
        self.connection_pool = self.get_mysql_connector_pool()

    def get_mysql_connector_pool(self) -> mysql.connector.pooling.MySQLConnectionPool:
        connection_pool = mysql.connector.pooling.MySQLConnectionPool(pool_name=self.pool_name, pool_size=5,
                                                                      **self.mysql_config)
        return connection_pool

    def safe_get_connection(self):
        try:
            connection = self.connection_pool.get_connection()
            return connection
        except Exception as e:
            print(f"Error getting database connection: {e}")
            self.connection_pool.add_connection()
            connection = self.connection_pool.get_connection()
            return connection

    def safe_close_connection(self, connection: mysql.connector.connection.MySQLConnection):
        try:
            connection.close()
        except Exception as e:
            print(f"Error closing the connection: {e}")
