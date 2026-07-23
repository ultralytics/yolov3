# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""utils/initialization."""

import contextlib
import threading

from ultralytics.utils import emojis


class TryExcept(contextlib.ContextDecorator):
    """A context manager and decorator for handling exceptions with optional custom messages."""

    def __init__(self, msg=""):
        """Initialize with an optional message prefixed to any caught exception when printed."""
        self.msg = msg

    def __enter__(self):
        """Enter the exception-handling block (no setup required)."""

    def __exit__(self, exc_type, value, traceback):
        """Print the message and exception on exit, suppressing the exception so execution continues."""
        if value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


def threaded(func):
    """Decorates a function to run in a separate thread, returning the thread object.

    Usage: @threaded.
    """

    def wrapper(*args, **kwargs):
        """Start the wrapped function in a daemon thread and return the started thread."""
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


def notebook_init(verbose=True):
    """Initializes notebook environment by checking hardware, software requirements, and cleaning up if in Colab."""
    print("Checking setup...")

    import os
    import shutil

    from utils.general import check_font, is_colab
    from utils.torch_utils import select_device  # imports

    check_font()

    import psutil

    if is_colab():
        shutil.rmtree("/content/sample_data", ignore_errors=True)  # remove colab /sample_data directory

    # System info
    display = None
    if verbose:
        gb = 1 << 30  # bytes to GiB (1024 ** 3)
        ram = psutil.virtual_memory().total
        total, _used, free = shutil.disk_usage("/")
        with contextlib.suppress(Exception):  # clear display if ipython is installed
            from IPython import display

            display.clear_output()
        s = f"({os.cpu_count()} CPUs, {ram / gb:.1f} GB RAM, {(total - free) / gb:.1f}/{total / gb:.1f} GB disk)"
    else:
        s = ""

    select_device(newline=False)
    print(emojis(f"Setup complete ✅ {s}"))
    return display
