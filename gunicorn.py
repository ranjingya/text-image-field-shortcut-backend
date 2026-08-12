bind = "0.0.0.0:5000"

workers = 1
worker_class = "gthread"
threads = 8

timeout = 660
graceful_timeout = 30
keepalive = 5

max_requests = 500
max_requests_jitter = 50

accesslog = "-"
errorlog = "-"
loglevel = "info"
capture_output = True
