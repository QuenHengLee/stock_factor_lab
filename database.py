import configparser
from utils.config import Config
import pymysql
import pandas as pd


class Database:
    def __init__(self):
        self._config = Config()
        self._db_data = self._config.get_database_config()
        # self.connection()

    """
    stock_index = {open, high, low, close, volume, market_capital}
    """

    # 建立與DB的連線
    def create_connection(self):
        # 檢查DB版本&連線成功
        try:
            config = configparser.ConfigParser()
            config.read("config.ini")
            config_host = config["database"]["host"]
            config_port = int(config["database"]["port"])
            config_user = config["database"]["user"]
            config_password = config["database"]["password"]
            config_db = config["database"]["db"]
            config_charset = config["database"]["charset"]

            db = pymysql.connect(
                host=config_host,
                port=config_port,
                user=config_user,
                passwd=config_password,
                db=config_db,
                charset=config_charset,
            )
            return db
        except Exception as e:
            print(e)
            print("無法連結資料庫")
            return e

    # 取的公司的開高低收(stock)
    def get_daily_stock(self):
        try:
            db = self.create_connection()
            cursor = db.cursor()
            # data = cursor.fetchone()
            # print('連線成功')

            # 選取台股(有帶入一些條件，避免數量過多)
            sql = " SELECT company_symbol,name,date,open,high,low,close,volume,market_capital \
                    FROM company RIGHT JOIN stock ON company.id = stock.company_id \
                    WHERE exchange_name = 'TWSE'\
                    AND company_symbol>8700 \
                    AND company_symbol<9000"
            # AND date > 2020-01-01"

            cursor.execute(sql)
            data = cursor.fetchall()
            columns = [
                "company_symbol",
                "name",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "market_capital",
            ]
            df = pd.DataFrame(data, columns=columns)
            # print(df)
            return df

        except Exception as e:
            print(e)
            print("無法執行SQL語法")
            return e

    # 取得公司的財報(factorvalue)
    def get_finance_report(self):
        try:
            db = self.create_connection()
            cursor = db.cursor()
            sql = " SELECT date, company_symbol, factor_name, factor_value \
                    FROM factor RIGHT JOIN factorvalue ON factor.id = factorvalue.factor_id  \
                    LEFT JOIN  company ON factorvalue.company_id = company.id \
                    WHERE exchange_name='TWSE'\
                    AND company_symbol>8700 \
                    AND company_symbol<9000"
            # AND date > 2020-01-01"

            cursor.execute(sql)
            data = cursor.fetchall()
            columns = ["date", "company_symbol", "factor_name", "factor_value"]
            df = pd.DataFrame(data, columns=columns)
            # print('The raw data get from database:\n')
            # print(df)
            return df

        except Exception as e:
            print(e)
            print("無法執行SQL語法")
            return e


if __name__ == "__main__":
    db = Database()
    db.get_finance_report()

    # db.select_index('open')
