from prometheus_client import Counter, Histogram


HTTP_LATENCY = Histogram(
    "rag_http_latency_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
)

HTTP_ERRORS = Counter(
    "rag_http_errors_total",
    "HTTP errors total",
    ["endpoint", "type"],
)

GENERATE_LATENCY = Histogram(
    "rag_generate_latency_seconds",
    "Generate endpoint latency in seconds",
    ["mode", "retrieval"],
)

GENERATE_ERRORS = Counter(
    "rag_generate_errors_total",
    "Generate endpoint errors total",
    ["mode", "retrieval", "type"],
)
