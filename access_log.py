import logging
from urllib.parse import unquote_plus


class AccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg != '%s - "%s %s HTTP/%s" %d' or not isinstance(record.args, tuple) or len(record.args) != 5:
            return True

        client, method, target, version, status = record.args
        path, separator, query = target.partition("?")
        if path == "/health" and 200 <= status < 300:
            return False

        if path == "/ask" and separator:
            parts = []
            for parameter in query.split("&"):
                name, equals, value = parameter.partition("=")
                if name == "q" and equals:
                    decoded = unquote_plus(value)
                    value = "".join(char if char.isprintable() else repr(char)[1:-1] for char in decoded)
                parts.append(name + equals + value)
            record.args = (client, method, path + separator + "&".join(parts), version, status)
        return True
