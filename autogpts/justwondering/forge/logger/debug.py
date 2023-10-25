import functools
from pygments import highlight, lexers, formatters
import json
import logging


CHAT = 29
logging.addLevelName(CHAT, "CHAT")

RESET_SEQ: str = "\033[0m"
COLOR_SEQ: str = "\033[1;%dm"
BOLD_SEQ: str = "\033[1m"
UNDERLINE_SEQ: str = "\033[04m"

ORANGE: str = "\033[33m"
YELLOW: str = "\033[93m"
WHITE: str = "\33[37m"
BLUE: str = "\033[34m"
LIGHT_BLUE: str = "\033[94m"
RED: str = "\033[91m"
GREY: str = "\33[90m"
GREEN: str = "\033[92m"

EMOJIS: dict[str, str] = {
    "DEBUG": "🐛",
    "INFO": "📝",
    "CHAT": "💬",
    "WARNING": "⚠️",
    "ERROR": "❌",
    "CRITICAL": "💥",
}

KEYWORD_COLORS: dict[str, str] = {
    "DEBUG": WHITE,
    "INFO": LIGHT_BLUE,
    "CHAT": GREEN,
    "WARNING": YELLOW,
    "ERROR": ORANGE,
    "CRITICAL": RED,
}


def formatter_message(message: str, use_color: bool = True) -> str:
    """
    Syntax highlight certain keywords
    """
    if use_color:
        message = message.replace(
            "$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


class ConsoleFormatter(logging.Formatter):
    """
    This Formatted simply colors in the levelname i.e 'INFO', 'DEBUG'
    """

    def __init__(
        self, fmt: str, datefmt: str = None, style: str = "%", use_color: bool = True
    ):
        super().__init__(fmt, datefmt, style)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """
        Format and highlight certain keywords
        """
        rec = record
        levelname = rec.levelname
        if self.use_color and levelname in KEYWORD_COLORS:
            levelname_color = KEYWORD_COLORS[levelname] + levelname + RESET_SEQ
            rec.levelname = levelname_color
        rec.name = f"{GREY}{rec.name:<15}{RESET_SEQ}"
        try:
            json_object = json.loads(rec.msg)
            if isinstance(json_object, (dict, list, str, int, float, bool, type(None))):
                pretty_message = json.dumps(json_object, indent=4)
                pretty_message = highlight(pretty_message, lexers.JsonLexer(),
                                           formatters.TerminalFormatter())
        except json.JSONDecodeError:
            pretty_message = rec.msg
        rec.msg = (
            KEYWORD_COLORS[levelname] + EMOJIS[levelname] +
            "  " + pretty_message + RESET_SEQ
        )
        return logging.Formatter.format(self, rec)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        record.msg = self.format_json(record.msg)
        return super().format(record)

    @staticmethod
    def format_json(message):
        try:
            json_object = json.loads(message)
            if isinstance(json_object, (dict, list, str, int, float, bool, type(None))):
                pretty_message = json.dumps(json_object, indent=4)
                return pretty_message
        except json.JSONDecodeError:
            pass
        return message


class PathLogger(logging.Logger):
    """
    This adds extra logging functions such as logger.trade and also
    sets the logger to use the custom formatter
    """
    # rewrite the console format with path and extract line where to jump to the code
    # CONSOLE_FORMAT: str = (
    #     "[%(asctime)s] [$BOLD%(name)-15s$RESET] [%(levelname)-8s]\t%(message)s"
    # )
    # ConsoleFormatterwithPath: str = (
    #     "[%(asctime)s] [$BOLD%(name)-15s$RESET] [%(levelname)-8s]\t%(message)s"
    # )

    ConsoleFormatterwithPath: str = "'[%(asctime)s] [%(pathname)s:%(lineno)d][%(name)s ] [%(levelname)s]-8s \n %(message)s"
    COLOR_FORMAT: str = formatter_message(ConsoleFormatterwithPath, True)

    def __init__(self, name: str, logLevel: str = "DEBUG"):
        logging.Logger.__init__(self, name, logLevel)
        console_formatter = ConsoleFormatter(self.COLOR_FORMAT)
        console = logging.StreamHandler()
        console.setFormatter(console_formatter)
        self.addHandler(console)


def log_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = PathLogger(__name__)
        logger.debug(f'Calling function {func.__name__}')
        result = func(*args, **kwargs)
        logger.debug(f'Function {func.__name__} returned {result}')
        return result
    return wrapper


def log_decorator(func):
    def wrapper(*args, **kwargs):
        logger = PathLogger(__name__)
        result = func(*args, **kwargs)
        json_str = json.dumps(result, indent=4)
        logger.debug(highlight(json_str, lexers.JsonLexer(),
                     formatters.TerminalFormatter()))
        return result
    return wrapper
