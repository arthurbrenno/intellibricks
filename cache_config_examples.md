```python
from datetime import timedelta

cache_config = CacheConfig(
    enabled=True,
    ttl=timedelta(minutes=10),
    cache_key='user_session_prompt'
)
```
