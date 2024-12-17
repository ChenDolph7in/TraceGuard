from myUtils.MysqlUtil import MysqlUtil


class AVMysqlUtil(MysqlUtil):
    def __init__(self, pool_name, mysql_config, save_image=False):
        super().__init__(pool_name, mysql_config)
        self.save_image = save_image
        self.create_table()

    def create_table(self):
        connection = self.safe_get_connection()
        cursor = connection.cursor()

        create_table_query = """
        CREATE TABLE IF NOT EXISTS vehicle_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            frame INT,
            simulation_time FLOAT,
            speed_kmh FLOAT,
            location_x FLOAT,
            location_y FLOAT,
            location_z FLOAT,
            velocity_x FLOAT,
            velocity_y FLOAT,
            velocity_z FLOAT,
            compass FLOAT,
            accelerometer_x FLOAT,
            accelerometer_y FLOAT,
            accelerometer_z FLOAT,
            gyroscope_x FLOAT,
            gyroscope_y FLOAT,
            gyroscope_z FLOAT,
            gnss_lat FLOAT,
            gnss_lon FLOAT,
            weather_cloudiness FLOAT,
            weather_precipitation FLOAT,
            weather_fog_density FLOAT,
            weather_fog_distance FLOAT,
            weather_wetness FLOAT,
            weather_wind_intensity FLOAT,
            weather_sun_altitude_angle FLOAT,
            weather_sun_azimuth_angle FLOAT,
            temperature FLOAT,
            collision FLOAT, 
            number_of_vehicles INT,
            city VARCHAR(50),  
            brand VARCHAR(50), 
            throttle FLOAT,
            steer FLOAT,
            brake FLOAT,
            reverse BOOLEAN,
            hand_brake BOOLEAN,
            manual_gear_shift BOOLEAN,
            gear INT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            {image_column}
        )
        """

        image_column = ", image MEDIUMBLOB" if self.save_image else ""
        create_table_query = create_table_query.format(image_column=image_column)

        try:
            cursor.execute(create_table_query)
            connection.commit()
        except Exception as e:
            print(f"Error creating saved_data table: {e}")
        finally:
            cursor.close()
            self.safe_close_connection(connection)

    def insert_data(self, data):
        connection = self.safe_get_connection()
        cursor = connection.cursor()

        if self.save_image:
            image_column = ", image"
            image_placeholder = ", %s"
        else:
            image_column = ""
            image_placeholder = ""

        insert_query = f"""
        INSERT INTO vehicle_data (
            frame, simulation_time, speed_kmh, 
            location_x, location_y, location_z, 
            velocity_x, velocity_y, velocity_z, 
            compass, accelerometer_x, accelerometer_y, accelerometer_z,
            gyroscope_x, gyroscope_y, gyroscope_z,
            gnss_lat, gnss_lon, 
            weather_cloudiness, weather_precipitation, 
            weather_fog_density, weather_fog_distance, 
            weather_wetness, weather_wind_intensity, 
            weather_sun_altitude_angle, weather_sun_azimuth_angle,
            temperature, collision, 
            number_of_vehicles,
            city, brand,
            throttle, steer, brake, reverse, hand_brake, 
            manual_gear_shift, gear
            {image_column}
        ) VALUES (%s, %s, %s, 
                  %s, %s, %s, 
                  %s, %s, %s, 
                  %s, %s, %s, %s,
                  %s, %s, %s,
                  %s, %s, 
                  %s, %s, 
                  %s, %s, 
                  %s, %s, 
                  %s, %s,
                  %s, %s, 
                  %s, %s,
                  %s, %s, %s, %s, %s, 
                  %s, %s, %s
                  {image_placeholder}
        )
        """

        params = (
                data['frame'],
                data['simulation_time'],
                data['speed_kmh'],
                data['location']['x'],
                data['location']['y'],
                data['location']['z'],
                data['velocity']['x'],
                data['velocity']['y'],
                data['velocity']['z'],
                data['compass'],
                data['accelerometer'][0],
                data['accelerometer'][1],
                data['accelerometer'][2],
                data['gyroscope'][0],
                data['gyroscope'][1],
                data['gyroscope'][2],
                data['gnss_lat'],
                data['gnss_lon'],
                data['weather']['cloudiness'],
                data['weather']['precipitation'],
                data['weather']['fog_density'],
                data['weather']['fog_distance'],
                data['weather']['wetness'],
                data['weather']['wind_intensity'],
                data['weather']['sun_altitude_angle'],
                data['weather']['sun_azimuth_angle'],
                data['temperature'],
                data['collision'],
                data['number_of_vehicles'],
                data['city'],
                data['brand'],
                data['throttle'],
                data['steer'],
                data['brake'],
                data['reverse'],
                data['hand_brake'],
                data['manual_gear_shift'],
                data['gear']
            )

        if self.save_image:
            params += (data['image'],)

        try:
            cursor.execute(insert_query, params)
            connection.commit()
        except Exception as e:
            print(f"Error adding data to database: {e}")
        finally:
            cursor.close()
            self.safe_close_connection(connection)

    def get_data(self, last_only=False, with_image=False):
        connection = self.safe_get_connection()
        cursor = connection.cursor(dictionary=True)
        try:
            select_query = "SELECT * FROM vehicle_data" if with_image and self.save_image else """
                SELECT id, frame, simulation_time, speed_kmh, location_x, location_y, 
                location_z, velocity_x, velocity_y, velocity_z, compass, accelerometer_x, 
                accelerometer_y, accelerometer_z, gyroscope_x, gyroscope_y, gyroscope_z, 
                gnss_lat, gnss_lon, weather_cloudiness, weather_precipitation, 
                weather_fog_density, weather_fog_distance, weather_wetness, 
                weather_wind_intensity, weather_sun_altitude_angle, 
                weather_sun_azimuth_angle, temperature, collision, number_of_vehicles, 
                city, brand, throttle, steer, brake, reverse, hand_brake, manual_gear_shift, 
                gear, timestamp FROM vehicle_data
                """

            if last_only:
                select_query += " ORDER BY id DESC LIMIT 30"
            cursor.execute(select_query)
            result = cursor.fetchall()
            return result
        except Exception as e:
            print(f"Error retrieving data from database: {e}")
            return None
        finally:
            cursor.close()
            self.safe_close_connection(connection)
