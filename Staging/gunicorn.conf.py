# gunicorn.conf.py

import multiprocessing

# Number of worker processes based on the number of CPU cores
workers = multiprocessing.cpu_count() * 2 + 1  # Formula to maximize CPU usage
threads = 4  # Increase threads to handle concurrent requests within each worker

# Maximum number of simultaneous clients
worker_connections = 1000  # Should be sufficient for your load

# Max number of pending requests before rejecting
backlog = 4096  # High backlog to handle traffic spikes

# Timeout for workers (in seconds)
timeout = 120  # Shorten the timeout to 90 seconds to prevent long-running requests from hogging resources

# Graceful timeout to give workers time to finish requests before restarting
graceful_timeout = 30  # Grace period for graceful worker restarts

# Logging
accesslog = '-'  # Log to stdout for better visibility
errorlog = '-'  # Log errors to stderr
loglevel = 'info'  # Log level set to 'info'

# Keep-alive settings
keepalive = 5  # Keep connection alive for up to 5 seconds for reuse

# Reloading the app automatically in case of changes (useful for development)
reload = False  # Set to True if you want auto-reload in development

# Maximum requests a worker can handle before being restarted (helps prevent memory leaks)
max_requests = 1000
max_requests_jitter = 100  # Randomly restart workers to balance load

# Limit the header size to prevent potential DoS attacks
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
