# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""utils/initialization."""

import contextlib
import platform
import threading


def emojis(str=""):
    """
    Returns a platform-dependent emoji-safe version of the input string, omitting emojis on Windows.

    Args:
        str (str): The input string that potentially contains emojis.

    Returns:
        str: The emoji-safe version of the input string, which omits emojis on Windows platforms and returns the original
             string on other platforms.

    Notes:
        On Windows systems, the function removes emoji characters to prevent display issues. On other platforms, it returns
        the string unchanged.

    Examples:
        ```python
        safe_string = emojis("Hello 😊")
        ```
    """
    return str.encode().decode("ascii", "ignore") if platform.system() == "Windows" else str


class TryExcept(contextlib.ContextDecorator):
    # YOLOv3 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager
    def __init__(self, msg=""):
        """
        Initializes the TryExcept context manager and decorator for exception handling with an optional custom message.

        Args:
            msg (str): Optional custom message to include in exception logs. Default is an empty string.

        Returns:
            None

        Examples:
            ```python
            # Using TryExcept as a decorator
            @TryExcept("An error occurred")
            def unsafe_function():
                return 1 / 0  # This will raise a ZeroDivisionError

            # Using TryExcept as a context manager
            with TryExcept("Operation failed"):
                result = 1 / 0  # This will also raise a ZeroDivisionError
            ```
        """
        self.msg = msg

    def __enter__(self):
        """
        Begin exception-handling block, optionally customizing the exception message when used with the TryExcept
        context manager.

        Returns:
            TryExcept: The TryExcept object itself, allowing exception handling within the context block.

        Examples:
            ```python
            with TryExcept("Custom error message"):
                # Your code here
                pass
            ```
        """
        pass

    def __exit__(self, exc_type, value, traceback):
        """
        Ends the exception-handling block, optionally prints a custom message with the exception, and suppresses
        exceptions within the context.

        Args:
          exc_type (type | None): The exception type being handled, or None if no exception occurred.
          value (Exception | None): The exception instance being handled, or None if no exception occurred.
          traceback (traceback | None): The traceback object associated with the exception, or None if no exception occurred.

        Returns:
          bool: Always returns True to suppress propagation of the exception.

        Note:
          Use this function as part of the TryExcept context manager or decorator to handle and log exceptions succinctly.
          ```python
          # Example usage as a decorator
          @TryExcept("Error during execution")
          def risky_function():
              # function code that might raise an exception
              pass

          # Example usage as a context manager
          with TryExcept("Error during scoped execution"):
              # code that might raise an exception within this block
              pass
          ```

        Refer to https://github.com/ultralytics/ultralytics for more details.
        """
        if value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


def threaded(func):
    """
    Decorates a function to run in a separate thread, returning the thread object.

    Args:
        func (Callable): Function to be decorated to run in a separate thread.

    Returns:
        Callable: The wrapper function that initiates the decorated function in a thread and returns the thread object.

    Example:
        ```python
        @threaded
        def example_function():
            print("This function is running in a separate thread.")

        thread = example_function()
        ```
    """

    def wrapper(*args, **kwargs):
        """
        Runs the decorated function in a separate thread and returns the thread object.

        Usage: @threaded.
        """
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


def join_threads(verbose=False):
    """
    Joins all daemon threads, excluding the main thread, with an optional verbose flag for logging.

    Args:
        verbose (bool): If True, prints the names of threads being joined. Default is False.

    Returns:
        None

    Notes:
        This function is typically used to ensure that all background threads complete their execution before the program
        exits.
    """
    main_thread = threading.current_thread()
    for t in threading.enumerate():
        if t is not main_thread:
            if verbose:
                print(f"Joining thread {t.name}")
            t.join()


def notebook_init(verbose=True):
    """
    Initializes the Jupyter notebook environment by performing hardware and software checks and cleaning up if in Google
    Colab.

    Args:
        verbose (bool): If True, prints detailed system information. Default is True.

    Returns:
        None

    Examples:
        ```python
        from ultralytics.utils.initialization import notebook_init

        notebook_init(verbose=True)
        ```
    """
    print("Checking setup...")

    import os
    import shutil

    from ultralytics.utils.checks import check_requirements

    from utils.general import check_font, is_colab
    from utils.torch_utils import select_device  # imports

    check_font()

    import psutil

    if check_requirements("wandb", install=False):
        os.system("pip uninstall -y wandb")  # eliminate unexpected account creation prompt with infinite hang
    if is_colab():
        shutil.rmtree("/content/sample_data", ignore_errors=True)  # remove colab /sample_data directory

    # System info
    display = None
    if verbose:
        gb = 1 << 30  # bytes to GiB (1024 ** 3)
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage("/")
        with contextlib.suppress(Exception):  # clear display if ipython is installed
            from IPython import display

            display.clear_output()
        s = f"({os.cpu_count()} CPUs, {ram / gb:.1f} GB RAM, {(total - free) / gb:.1f}/{total / gb:.1f} GB disk)"
    else:
        s = ""

    select_device(newline=False)
    print(emojis(f"Setup complete ✅ {s}"))
    return display
