import os
import json
import redis

class RedisCache:
    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            self.client = redis.Redis.from_url(url=redis_url,socket_timeout=2)
            self.client.ping()
        except redis.ConnectionError:
            self.client = None

    def get(self, key:str):
        if not self.client: return None
        try:
            val = self.client.get(key)
            return json.loads(val.decode('utf-8'))
        except Exception:
            return None

    def set(self, key:str, value:str, ttl:int = 600):
        if not self.client: return
        try:
            self.client.setex(key, ttl, json.dumps(value))
        except Exception:
            pass
            
        
        
        
            