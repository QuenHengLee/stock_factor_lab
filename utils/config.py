from configparser import ConfigParser

class Config():

    def __init__(self):
        self._parser = ConfigParser()
        self._parser.read("config.ini", encoding="utf-8")
        
    def get_database_config(self):
        database_config = {}
        database_config["host"]     = str(self._parser["database"]["host"])
        database_config["port"]     = int(self._parser["database"]["port"])
        database_config["user"]     = str(self._parser["database"]["user"])
        database_config["password"] = str(self._parser["database"]["password"])
        database_config["db"]       = str(self._parser["database"]["db"])
        database_config["charset"]  = str(self._parser["database"]["charset"])
        return database_config