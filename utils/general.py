# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""General utils."""

import contextlib
import glob
import inspect
import logging
import logging.config
import math
import os
import platform
import random
import re
import signal
import subprocess
import sys
import time
import urllib
from copy import deepcopy
from datetime import datetime
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from tarfile import is_tarfile
from typing import Optional
from zipfile import ZipFile, is_zipfile

import cv2
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml
from ultralytics.utils.checks import check_requirements

from utils import TryExcept, emojis
from utils.downloads import curl_download, gsutil_getsize
from utils.metrics import box_iou, fitness

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv3 root directory
RANK = int(os.getenv("RANK", -1))

# Settings
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv3 multiprocessing threads
DATASETS_DIR = Path(os.getenv("YOLOv5_DATASETS_DIR", ROOT.parent / "datasets"))  # global datasets directory
AUTOINSTALL = str(os.getenv("YOLOv5_AUTOINSTALL", True)).lower() == "true"  # global auto-install mode
VERBOSE = str(os.getenv("YOLOv5_VERBOSE", True)).lower() == "true"  # global verbose mode
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format
FONT = "Arial.ttf"  # https://ultralytics.com/assets/Arial.ttf

torch.set_printoptions(linewidth=320, precision=5, profile="long")
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)  # NumExpr max threads
os.environ["OMP_NUM_THREADS"] = "1" if platform.system() == "darwin" else str(NUM_THREADS)  # OpenMP (PyTorch and SciPy)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress verbose TF compiler warnings in Colab


def is_ascii(s=""):
    """
    Checks if the input string `s` is composed solely of ASCII characters; compatible with pre-Python 3.7 versions.

    Args:
        s (str): The input string to be checked for ASCII characters. Defaults to an empty string.

    Returns:
        bool: True if all characters in the string `s` are ASCII, otherwise False.

    Examples:
        ```python
        is_ascii("Hello World!")  # Returns: True
        is_ascii("こんにちは世界")  # Returns: False
        ```

    Notes:
        The function converts various input types to a string using `str(s)` to ensure compatibility with different input
        types like list, tuple, None, etc. before checking for ASCII characters.

    Links:
        For more information on ASCII characters: https://en.wikipedia.org/wiki/ASCII
    """
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode("ascii", "ignore")) == len(s)


def is_chinese(s="人工智能"):
    """
    Determines if the input string contains any Chinese characters.

    Args:
        s (str, optional): The string to be evaluated. Defaults to "人工智能".

    Returns:
        bool: True if the string contains any Chinese characters, False otherwise.

    Examples:
        ```python
        from ultralytics.utils import is_chinese

        print(is_chinese("测试"))  # True
        print(is_chinese("test"))  # False
        ```

    Note:
        This function is useful for text preprocessing in machine learning applications where handling multilingual
        text data is required.
    """
    return bool(re.search("[\u4e00-\u9fff]", str(s)))


def is_colab():
    """
    Checks if the current environment is a Google Colab instance.

    Returns:
        bool: True if running in a Google Colab environment, otherwise False.

    Notes:
        This function can be useful to conditionally execute code that should only run in Colab environments. For example, adjusting configurations that are specific to the Colab setup or utilizing Colab-specific functionalities.
    """
    return "google.colab" in sys.modules


def is_jupyter():
    """
    Check if the current script is running inside a Jupyter Notebook.

    Returns:
        bool: True if running inside a Jupyter Notebook, False otherwise.

    Notes:
        This function verifies if the environment is Jupyter-based, compatible with various platforms like Colab,
        Jupyterlab, Kaggle, Paperspace.

    Examples:
        ```python
        if is_jupyter():
            print("Running inside a Jupyter Notebook")
        else:
            print("Not running inside a Jupyter Notebook")
        ```
    """
    with contextlib.suppress(Exception):
        from IPython import get_ipython

        return get_ipython() is not None
    return False


def is_kaggle():
    """
    Determines if the environment is a Kaggle Notebook by checking environment variables.

    Returns:
        bool: True if running inside a Kaggle Notebook, False otherwise.

    Examples:
        ```python
        if is_kaggle():
            print("Running in a Kaggle Notebook")
        else:
            print("Not running in a Kaggle Notebook")
        ```
    """
    return os.environ.get("PWD") == "/kaggle/working" and os.environ.get("KAGGLE_URL_BASE") == "https://www.kaggle.com"


def is_docker() -> bool:
    """
    Check if the process runs inside a docker container.

    Returns:
        bool: True if the process is running inside a docker container, False otherwise.

    Examples:
        ```python
        if is_docker():
            print("Running in a docker container")
        else:
            print("Not running in a docker container")
        ```
    """
    if Path("/.dockerenv").exists():
        return True
    try:  # check if docker is in control groups
        with open("/proc/self/cgroup") as file:
            return any("docker" in line for line in file)
    except OSError:
        return False


def is_writeable(dir, test=False):
    """
    Determines if a directory is writeable, optionally tests by writing a file if `test=True`.

    Args:
        dir (str | Path): The directory path to check for write permissions.
        test (bool, optional): If True, attempts to write a temporary file to `dir` to test writeability. Defaults to False.

    Returns:
        bool: True if the directory is writeable, False otherwise. If `test=True`, returns True if a temporary file can
        be successfully written and deleted in the specified directory, False otherwise.

    Notes:
        - Using `os.access(dir, os.W_OK)` to check write permissions may not be reliable on Windows due to user access control
        (UAC) or other permission-related issues. To mitigate this, use the `test` parameter to perform an actual write test.
        - When `test=True`, a file named "tmp.txt" will be created and deleted in the specified directory.

    Examples:
        ```python
        from pathlib import Path

        # Check if directory is writable without test
        is_writeable(str(Path.home()))

        # Check if directory is writable with write test
        is_writeable(str(Path.home()), test=True)
        ```
    """
    if not test:
        return os.access(dir, os.W_OK)  # possible issues on Windows
    file = Path(dir) / "tmp.txt"
    try:
        with open(file, "w"):  # open file with write permissions
            pass
        file.unlink()  # remove file
        return True
    except OSError:
        return False


LOGGING_NAME = "yolov5"


def set_logging(name=LOGGING_NAME, verbose=True):
    """
    Configures logging for the Ultralytics library.

    Args:
        name (str): The name identifier for the logger. Default is 'yolov5'.
        verbose (bool): If True, sets logging level to INFO, otherwise sets to ERROR. Default is True.

    Returns:
        None

    Notes:
        This function is designed to adjust the logging configuration based on the environment and specified verbosity.
        It sets different logging levels based on whether the code is running under multi-GPU training conditions.
    """
    rank = int(os.getenv("RANK", -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {name: {"format": "%(message)s"}},
            "handlers": {
                name: {
                    "class": "logging.StreamHandler",
                    "formatter": name,
                    "level": level,
                }
            },
            "loggers": {
                name: {
                    "level": level,
                    "handlers": [name],
                    "propagate": False,
                }
            },
        }
    )


set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)
if platform.system() == "Windows":
    for fn in LOGGER.info, LOGGER.warning:
        setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))  # emoji safe logging


def user_config_dir(dir="Ultralytics", env_var="YOLOV5_CONFIG_DIR"):
    """
    Returns user configuration directory path, preferring `env_var` if set, or an OS-specific path; creates directory if
    needed.

    Args:
        dir (str): Name of the directory to be created or used within the user configuration directory. Defaults to "Ultralytics".
        env_var (str): Environment variable name to use for obtaining the configuration directory path. Defaults to "YOLOV5_CONFIG_DIR".

    Returns:
        Path: User configuration directory path (PosixPath | WindowsPath).

    Notes:
        - On Windows, the default path is "AppData/Roaming".
        - On Linux, the default path is ".config".
        - On macOS (Darwin), the default path is "Library/Application Support".
        - If the determined OS-specific directory is not writable, the function defaults to using "/tmp".
    """
    env = os.getenv(env_var)
    if env:
        path = Path(env)  # use environment variable
    else:
        cfg = {"Windows": "AppData/Roaming", "Linux": ".config", "Darwin": "Library/Application Support"}  # 3 OS dirs
        path = Path.home() / cfg.get(platform.system(), "")  # OS-specific config dir
        path = (path if is_writeable(path) else Path("/tmp")) / dir  # GCP and AWS lambda fix, only /tmp is writeable
    path.mkdir(exist_ok=True)  # make if required
    return path


CONFIG_DIR = user_config_dir()  # Ultralytics settings dir


class Profile(contextlib.ContextDecorator):
    # YOLOv3 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0):
        """
        Initializes a profiling context for YOLOv3 with an optional timing threshold.

        Args:
            t (float, optional): A timing threshold value in seconds. If the total time of the profiled block exceeds
                                 this threshold, a warning or log message may be triggered. Default is 0.0.

        Returns:
            None
        Notes:
            - This class can be used as a decorator with `@Profile()` or within a `with Profile():` context manager.
            - CUDA availability is checked during initialization to ensure profiling takes GPU acceleration into account.
        """
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        """
        Starts the profiling timer, returning the profile instance for use with the `@Profile` decorator or the `with
        Profile():` context manager.

        Returns:
            Profile: The current instance with the profiling timer started.

        Example:
            ```python
            with Profile() as p:
                # code to be profiled
            ```

        Notes:
            - This method should be used in conjunction with `__exit__` to measure elapsed time.
            - The `time` method invoked by this function is defined elsewhere in the Profile class.
        """
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        """
        Ends profiling, calculating the elapsed time since profiling started and updating the total time accumulated
        within the context.

        Args:
            type (Optional[type]): The exception class being handled (if any).
            value (Optional[BaseException]): The exception instance being handled (if any).
            traceback (Optional[traceback]): The traceback object providing details about the exception (if any).

        Returns:
            bool: True if the exception was handled, False otherwise.
        """
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        """
        Returns current time, ensuring CUDA operations are synchronized if on GPU.

        Returns:
            float: The current time in seconds, synchronized with CUDA operations if a CUDA-enabled GPU is available.

        Notes:
            This method synchronizes GPU operations to ensure an accurate time reading for CUDA-enabled environments. This is
            essential for profiling code that includes CUDA operations to get precise timing metrics.

        Example:
            ```python
            profile = Profile()
            with profile:
                # Your code block
                pass
            print(f"Elapsed time: {profile.dt} seconds")
            ```
        """
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()


class Timeout(contextlib.ContextDecorator):
    # YOLOv3 Timeout class. Usage: @Timeout(seconds) decorator or 'with Timeout(seconds):' context manager
    def __init__(self, seconds, *, timeout_msg="", suppress_timeout_errors=True):
        """
        Initializes a timeout context manager or decorator for enforcing execution time limits.

        Args:
            seconds (int): The maximum allowed time for the block of code or function execution, in seconds.
            timeout_msg (str, optional): Custom message to display if the timeout is reached. Defaults to an empty string.
            suppress_timeout_errors (bool, optional): Whether to suppress timeout errors or propagate them. Defaults to True.

        Returns:
            None

        Examples:
            ```python
            from ultralytics.utils.general import Timeout

            # Using as a context manager
            with Timeout(5, timeout_msg="Operation timed out!"):
                perform_long_running_task()

            # Using as a decorator
            @Timeout(10, timeout_msg="Function execution exceeded time limit!")
            def my_function():
                perform_another_long_running_task()
            ```
        """
        self.seconds = int(seconds)
        self.timeout_message = timeout_msg
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        """
        Raises:
            TimeoutError: Indicates that the operation has exceeded the specified time limit, including a custom message if provided.
        
        Args:
            signum (int): Signal number received, typically a SIGALRM.
            frame (Optional[FrameType]): Current stack frame (usually not used but required by the signal handler).
        
        Returns:
            None: This function does not return a value.
        
        Notes:
            This function is designed to be used internally within the `Timeout` class for handling timeout signals.
            It is automatically invoked when the specified timeout duration elapses.
        """
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        """
        __enter__()

        Starts the countdown for the timeout context or decorator, setting up a signal alarm to be triggered after the specified duration.

        Returns:
            Timeout: The current instance of the Timeout class.

        Raises:
            TimeoutError: If the specified timeout duration elapses while the context is active.
        """
        if platform.system() != "Windows":  # not supported on Windows
            signal.signal(signal.SIGALRM, self._timeout_handler)  # Set handler for SIGALRM
            signal.alarm(self.seconds)  # start countdown for SIGALRM to be raised

    def __exit__(self, exc_type, exc_val, exc_tb):
        """__exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[object]) ->
        None:
        """Handles exit from the timeout context, suppressing TimeoutError if specified.
        
            Args:
                exc_type (Optional[type]): The exception type, if any, that caused the exit.
                exc_val (Optional[BaseException]): The exception instance, if any, that caused the exit.
                exc_tb (Optional[object]): The traceback object, if any, for the exception that caused the exit.
        
            Returns:
                None: This method does not return a value.
            """
            if platform.system() != "Windows":
                signal.alarm(0)  # Cancel SIGALRM if it's scheduled
                if self.suppress and exc_type is TimeoutError:  # Suppress TimeoutError
                    return True  # Suppress exception (TimeoutError)
        """
        if platform.system() != "Windows":
            signal.alarm(0)  # Cancel SIGALRM if it's scheduled
            if self.suppress and exc_type is TimeoutError:  # Suppress TimeoutError
                return True


class WorkingDirectory(contextlib.ContextDecorator):
    # Usage: @WorkingDirectory(dir) decorator or 'with WorkingDirectory(dir):' context manager
    def __init__(self, new_dir):
        """
        Initializes a context manager to temporarily change the working directory, reverting it upon exit.

        Args:
            new_dir (str | Path): The directory to which the working directory will be temporarily changed.

        Returns:
            None

        Notes:
            - This class can be used as a decorator with `@WorkingDirectory(dir)` or within a `with WorkingDirectory(dir):`
              context block to ensure the code runs inside the specified directory context.
        """
        self.dir = new_dir  # new dir
        self.cwd = Path.cwd().resolve()  # current dir

    def __enter__(self):
        """
        Temporarily changes the current working directory to `new_dir`, reverting to the original on exit.

        Returns:
            Path: The new working directory as a Path object.

        Examples:
            Temporarily change the working directory:

            ```python
            with WorkingDirectory('/path/to/new_directory'):
                # Execute code within this directory context
            ```
        """
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """__exit__(self, exc_type, exc_val, exc_tb):"""
            Reverts to the original working directory upon exiting the context manager.
        
            Args:
                exc_type (type | None): The exception type if an exception was raised, otherwise None.
                exc_val (Exception | None): The exception instance if an exception was raised, otherwise None.
                exc_tb (traceback | None): The traceback object if an exception was raised, otherwise None.
        
            Returns:
                bool: True if the exception is handled, otherwise False.
            
            Example:
                with WorkingDirectory('/tmp'):
                    # code execution within /tmp directory
                # automatically reverts to original directory after the context
            """
            os.chdir(self.cwd)
            return False  # Do not suppress exceptions
        """
        os.chdir(self.cwd)


def methods(instance):
    """
    Returns a list of callable methods for a given class or instance, excluding magic methods.

    Args:
        instance (object): The class or instance to inspect for callable methods.

    Returns:
        list[str]: A list of method names that are callable and not magic methods (do not start and end with double underscores).

    Examples:
        ```python
        class MyClass:
            def method1(self):
                pass

            def method2(self):
                pass

            def __magic_method__(self):
                pass

        instance = MyClass()
        print(methods(instance))  # Output: ['method1', 'method2']
        ```
    """
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    """
    Prints the arguments of the calling function, including optional file and function names.

    Args:
        args (dict | None): A dictionary of arguments to print. If None, arguments are automatically retrieved from
                            the calling function's frame.
        show_file (bool): Whether to display the filename of the calling function (default: True).
        show_func (bool): Whether to display the function name of the calling function (default: False).

    Returns:
        None

    Notes:
        This utility function is helpful for debugging and logging purposes, providing insights into the
        execution context and parameters passed to various functions within the codebase.
    """
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        file = Path(file).stem
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    LOGGER.info(colorstr(s) + ", ".join(f"{k}={v}" for k, v in args.items()))


def init_seeds(seed=0, deterministic=False):
    """
    Initializes random number generator (RNG) seeds for reproducibility, optionally enforcing deterministic behavior.

    Args:
        seed (int): RNG seed to ensure reproducibility. Defaults to 0.
        deterministic (bool): If True, enforces deterministic behavior and sets `torch` backend configurations appropriately.
            Defaults to False.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    # torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    if deterministic and check_version(torch.__version__, "1.12.0"):  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)


def intersect_dicts(da, db, exclude=()):
    """
    Intersects two dictionaries by matching keys and shapes, excluding specified keys, and retains values from the first
    dictionary.

    Args:
        da (dict): The first dictionary whose values will be retained in the intersection.
        db (dict): The second dictionary used to find matching keys and shapes.
        exclude (tuple[list[str]], optional): A tuple of keys to be excluded from the intersection. Default is ().

    Returns:
        dict: A dictionary containing the intersected key-value pairs from `da` where keys and shapes match those in `db`, except for the excluded keys.

    Examples:
        ```python
        da = {'a': torch.randn(1, 2), 'b': torch.randn(1, 3), 'c': torch.randn(2, 2)}
        db = {'a': torch.randn(1, 2), 'b': torch.randn(1, 4), 'd': torch.randn(2, 2)}
        result = intersect_dicts(da, db)
        # Output: {'a': tensor([..])}

        result = intersect_dicts(da, db, exclude=('a',))
        # Output: {}
        ```

    Notes:
        - A key-value pair from `da` is included in the result only if there exists a key in `db` with the same shape.
        - Any keys listed in `exclude` are not included in the output, even if they have matching shapes.
    """
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def get_default_args(func):
    """
    Retrieves a dictionary of a function's default arguments using the inspect module.

    Args:
        func (function): The target function from which to extract default arguments.

    Returns:
        dict: A dictionary where keys are argument names and values are the respective default values.

    Examples:
        ```python
        def sample_function(a, b=10, c="default"):
            pass

        default_args = get_default_args(sample_function)
        print(default_args)  # Output: {'b': 10, 'c': 'default'}
        ```
    """
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_latest_run(search_dir="."):
    """
    Returns path to the most recent 'last.pt' file within the specified directory.

    Args:
        search_dir (str): The directory in which to search for the latest 'last.pt' file. Defaults to the current
            directory.
    
    Returns:
        str: The path to the most recent 'last.pt' file if found, otherwise an empty string.
    
    Examples:
        >>> get_latest_run("/path/to/runs")
        '/path/to/runs/exp2/last.pt'
        
    Notes:
        - The function performs a recursive search within the provided directory.
        - If no 'last.pt' files are found, the function returns an empty string.
    """
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""


def file_age(path=__file__):
    """
    Returns the number of days since the last update of the file specified by 'path'.

    Args:
        path (str | Path, optional): The path to the file. Defaults to the current script file (__file__).

    Returns:
        float: Number of days since the last modification to the file.
    """
    dt = datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime)  # delta
    return dt.days  # + dt.seconds / 86400  # fractional days


def file_date(path=__file__):
    """
    Returns file modification date in 'YYYY-M-D' format for the file at 'path'.

    Args:
        path (str | Path, optional): The file path to check. Defaults to the current file (__file__).

    Returns:
        str: The file modification date in 'YYYY-M-D' format.

    Example:
        ```python
        file_date("/path/to/your/file.py")
        # '2023-3-25'
        ```

    Note:
        Use pathlib.Path for cross-platform file path compatibility.
    """
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f"{t.year}-{t.month}-{t.day}"


def file_size(path):
    """
    Returns the size of a file or total size of files in a directory at the given path in MB.

    Args:
        path (str | Path): The filesystem path to a file or directory.

    Returns:
        float: The size in megabytes (MB) of the file or the total size of all files in the directory.

    Examples:
        Calculate the size of a single file:
        ```python
        size = file_size('/path/to/file.txt')
        print(f"File size: {size} MB")
        ```

        Calculate the total size of all files in a directory:
        ```python
        size = file_size('/path/to/directory')
        print(f"Directory size: {size} MB")
        ```

    Notes:
        - The size computation converts bytes to megabytes (MB) using the factor 1 MB = 1024 * 1024 bytes.
        - The function returns 0.0 if the specified path is neither a file nor a directory.
    """
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / mb
    else:
        return 0.0


def check_online():
    """
    Checks internet connectivity by attempting to connect to "1.1.1.1" on port 443 twice; returns True if successful.

    Returns:
        bool: True if internet connection is available, False otherwise.

    Examples:
        ```python
        from ultralytics import check_online

        connected = check_online()
        if connected:
            print("Internet connection is available.")
        else:
            print("No internet connection.")
        ```
    """
    import socket

    def run_once():
        """Attempts a single internet connectivity check to '1.1.1.1' on port 443 and returns True if successful."""
        try:
            socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
            return True
        except OSError:
            return False

    return run_once() or run_once()  # check twice to increase robustness to intermittent connectivity issues


def git_describe(path=ROOT):  # path must be a directory
    """
    Returns human-readable git description of a directory if it's a git repository, otherwise an empty string.

    Args:
        path (str | Path, optional): The directory path to check for git description. Defaults to ROOT.

    Returns:
        str: A git description string if the directory is a git repository, otherwise an empty string.

    Raises:
        AssertionError: If the provided `path` does not contain a '.git' directory.

    Examples:
        ```python
        from ultralytics.utils.general import git_describe
        print(git_describe())  # Example output: 'v0.1.0-20-gabcdef123'
        print(git_describe("/path/to/repo"))  # Example output: 'v0.1.0-20-gabcdef123'
        ```

    Notes:
        The function runs the git command 'git describe --tags --long --always' and requires that the `path` points to a
        valid git repository directory.
    """
    try:
        assert (Path(path) / ".git").is_dir()
        return check_output(f"git -C {path} describe --tags --long --always", shell=True).decode()[:-1]
    except Exception:
        return ""


@TryExcept()
@WorkingDirectory(ROOT)
def check_git_status(repo="ultralytics/yolov5", branch="master"):
    """
    Checks the status of the local git repository against the remote repository and provides suggestions to update if
    necessary.

    Args:
        repo (str): The GitHub repository to check against. Defaults to "ultralytics/yolov5".
        branch (str): The branch of the remote repository to compare with. Defaults to "master".

    Returns:
        None

    Notes:
        - Requires an active internet connection and git to be installed on the local system.
        - Asserts that the local directory is a git repository and the system is online before proceeding with checks.
        - If the repository is found to be behind the remote, it suggests using `git pull` to update the repository.
        - For more information or updates, visit the repository URL: https://github.com/ultralytics/yolov5.

    Examples:
        ```python
        check_git_status()
        ```

        This will check if the local git repository of `ultralytics/yolov5` on the `master` branch is up-to-date with its remote counterpart.
    """
    url = f"https://github.com/{repo}"
    msg = f", for updates see {url}"
    s = colorstr("github: ")  # string
    assert Path(".git").exists(), s + "skipping check (not a git repository)" + msg
    assert check_online(), s + "skipping check (offline)" + msg

    splits = re.split(pattern=r"\s", string=check_output("git remote -v", shell=True).decode())
    matches = [repo in s for s in splits]
    if any(matches):
        remote = splits[matches.index(True) - 1]
    else:
        remote = "ultralytics"
        check_output(f"git remote add {remote} {url}", shell=True)
    check_output(f"git fetch {remote}", shell=True, timeout=5)  # git fetch
    local_branch = check_output("git rev-parse --abbrev-ref HEAD", shell=True).decode().strip()  # checked out
    n = int(check_output(f"git rev-list {local_branch}..{remote}/{branch} --count", shell=True))  # commits behind
    if n > 0:
        pull = "git pull" if remote == "origin" else f"git pull {remote} {branch}"
        s += f"⚠️ YOLOv3 is out of date by {n} commit{'s' * (n > 1)}. Use '{pull}' or 'git clone {url}' to update."
    else:
        s += f"up to date with {url} ✅"
    LOGGER.info(s)


@WorkingDirectory(ROOT)
def check_git_info(path="."):
    """
    Checks YOLOv3 git information including remote URL, current branch, and latest commit hash in the specified path.

    Args:
        path (str): Path to the directory containing the git repository. Defaults to current directory ".".

    Returns:
        dict: A dictionary containing:
            - 'remote' (str): URL of the remote repository.
            - 'branch' (str | None): Current branch name or None if in 'detached HEAD' state.
            - 'commit' (str): Hexadecimal hash of the latest commit.

    Raises:
        git.exc.InvalidGitRepositoryError: If the specified path does not contain a valid git repository.

    Notes:
        Requires 'gitpython' library to be installed.

    Example:
        ```python
        git_info = check_git_info("/path/to/repo")
        print(git_info["remote"])  # Output: 'https://github.com/ultralytics/yolov5'
        print(git_info["branch"])  # Output: 'main'
        print(git_info["commit"])  # Output: '3134699c73af83aac2a481435550b968d5792c0d'
        ```
    """
    check_requirements("gitpython")
    import git

    try:
        repo = git.Repo(path)
        remote = repo.remotes.origin.url.replace(".git", "")  # i.e. 'https://github.com/ultralytics/yolov5'
        commit = repo.head.commit.hexsha  # i.e. '3134699c73af83aac2a481435550b968d5792c0d'
        try:
            branch = repo.active_branch.name  # i.e. 'main'
        except TypeError:  # not on any branch
            branch = None  # i.e. 'detached HEAD' state
        return {"remote": remote, "branch": branch, "commit": commit}
    except git.exc.InvalidGitRepositoryError:  # path is not a git dir
        return {"remote": None, "branch": None, "commit": None}


def check_python(minimum="3.7.0"):
    """
    Checks if the current Python version meets the specified minimum requirement, raising an error if it does not.

    Args:
        minimum (str): Minimum required Python version specified as a string (e.g., "3.7.0").

    Returns:
        None: This function does not return a value. It either completes successfully or raises an AssertionError.

    Raises:
        AssertionError: If the current Python version is lower than the specified minimum.

    Notes:
        - This function ensures compatibility by enforcing a minimum Python version.
        - Edit the minimum variable as needed to adapt to different version requirements.

    Examples:
        ```python
        check_python(minimum="3.8.0")  # Ensures running Python version is at least 3.8.0
        ```

    See Also:
        - [Python version tuple](https://docs.python.org/3/library/sys.html#sys.version_info) for more details on querying the Python version.
    """
    check_version(platform.python_version(), minimum, name="Python ", hard=True)


def check_version(current="0.0.0", minimum="0.0.0", name="version ", pinned=False, hard=False, verbose=False):
    """
    Verifies that the current version of a specified package meets or exceeds a minimum version requirement.

    Args:
        current (str): The current version string of the package to be checked.
        minimum (str): The minimum required version string.
        name (str): The name of the package (default is "version ").
        pinned (bool): If True, requires the current version to be exactly equal to the minimum version. Defaults to False.
        hard (bool): If True, raises an assertion error if the version check fails. Defaults to False.
        verbose (bool): If True, logs a warning if the version check fails. Defaults to False.

    Returns:
        bool: True if the current version meets or exceeds the minimum version requirement, False otherwise.

    Example:
        ```python
        check_version(current="1.7.0", minimum="1.6.0", name="numpy ")
        ```

    Notes:
        - Utilizes `pkg_resources.parse_version` for comparing version strings.
        - Raises an error with an emoji-laden message if the `hard` parameter is True and the check fails.
        - Logs a warning message if the version check fails and `verbose` is True, without raising an error.
    """
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f"WARNING ⚠️ {name}{minimum} is required by YOLOv3, but {name}{current} is currently installed"  # string
    if hard:
        assert result, emojis(s)  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(s)
    return result


def check_img_size(imgsz, s=32, floor=0):
    """
    Adjusts image size to be divisible by a specified value, ensuring it stays above a minimum floor value.

    Args:
        imgsz (int | list[int]): Desired image size. Can be an integer specifying a single dimension or a list specifying
            multiple dimensions.
        s (int): Stride value to ensure the image size is divisible by. Default is 32.
        floor (int): Minimum allowable dimension value. Default is 0.

    Returns:
        int | list[int]: Adjusted image size(s) that are divisible by `s` and above the `floor` value. If the input was a
            single dimension (integer), the return value is an integer. If the input was multiple dimensions (list), the
            return value is a list.

    Notes:
        Logs a warning if the input image size was adjusted to meet the divisibility or floor constraints.

    Example:
        ```python
        # Single dimension image size
        adjusted_size = check_img_size(641)  # Adjusted to 640 (divisible by 32)

        # Multiple dimensions image size
        adjusted_sizes = check_img_size([640, 481])  # Adjusted to [640, 480] (each divisible by 32)
        ```
    """
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f"WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}")
    return new_size


def check_imshow(warn=False):
    """
    Checks if the environment supports image display; warns if `warn=True` and display is unsupported.

    Args:
        warn (bool): If True, logs a warning message when image display is unsupported. Default is False.

    Returns:
        bool: True if the environment supports image display, False otherwise.

    Notes:
        This function uses `cv2.imshow` to test the display capability. It is designed to return False in Jupyter
        Notebooks, Docker Containers, and other non-GUI environments.

    Examples:
        ```python
        if check_imshow(warn=True):
            cv2.imshow('image', img)
            cv2.waitKey(0)
        else:
            print("Image display is not supported in this environment.")
        ```

    References:
        - https://github.com/ultralytics/ultralytics
    """
    try:
        assert not is_jupyter()
        assert not is_docker()
        cv2.imshow("test", np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f"WARNING ⚠️ Environment does not support cv2.imshow() or PIL Image.show()\n{e}")
        return False


def check_suffix(file="yolov5s.pt", suffix=(".pt",), msg=""):
    """
    Checks if file(s) have acceptable suffix(es).

    Args:
        file (str | list[str] | tuple[str]): Filename(s) to check.
        suffix (str | tuple[str]): Acceptable suffix(es) for the file(s). Can be a single suffix string or a tuple of suffixes.
        msg (str): Additional message to include in the exception if the suffix check fails.

    Returns:
        None

    Raises:
        AssertionError: If any file does not have one of the acceptable suffixes.

    Examples:
        ```python
        check_suffix("model.pt", suffix=(".pt", ".pth"))
        check_suffix(["model1.pt", "model2.onnx"], suffix=(".pt", ".onnx"))
        ```
    Notes:
        This function is particularly useful for validating model file types and ensuring compatibility.
        Ensure the `suffix` argument includes a dot, e.g., '.pt' rather than 'pt'.
    """
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def check_yaml(file, suffix=(".yaml", ".yml")):
    """
    Checks for valid YAML file with extensions '.yaml' or '.yml', optionally downloading the file if it doesn't exist.

    Args:
        file (str): Path to the YAML file or url to be checked.
        suffix (str | tuple[str], optional): Valid suffixes for the YAML file. Defaults to (".yaml", ".yml").

    Returns:
        str: Path to the valid YAML file.

    Raises:
        AssertionError: If file does not have valid suffix or if there are any issues with ensuring the file's existence.

    Example:
        ```python
        yaml_path = check_yaml("config.yaml")
        ```

    Note:
        Supports URLs and local file paths. Validates the file suffix and ensures file existence, downloading if necessary.
    """
    return check_file(file, suffix)


def check_file(file, suffix=""):
    """
    Checks for the existence of a file locally, downloads it if provided as a URL, supports ClearML dataset IDs, and
    optionally enforces a file suffix.

    Args:
        file (str): Path to the file to be checked.
        suffix (str | tuple | list): Optional. Allowed file suffix(es). Defaults to an empty string.

    Returns:
        str: Path to the verified file.

    Example:
        ```python
        file_path = check_file("https://example.com/model.pt", suffix=".pt")
        print(file_path)  # Outputs: model.pt
        ```
    """
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if os.path.isfile(file) or not file:  # exists
        return file
    elif file.startswith(("http:/", "https:/")):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file).split("?")[0]).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if os.path.isfile(file):
            LOGGER.info(f"Found {url} locally at {file}")  # file already exists
        else:
            LOGGER.info(f"Downloading {url} to {file}...")
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f"File download failed: {url}"  # check
        return file
    elif file.startswith("clearml://"):  # ClearML Dataset ID
        assert (
            "clearml" in sys.modules
        ), "ClearML is not installed, so cannot use ClearML dataset. Try running 'pip install clearml'."
        return file
    else:  # search
        files = []
        for d in "data", "models", "utils":  # search directories
            files.extend(glob.glob(str(ROOT / d / "**" / file), recursive=True))  # find file
        assert len(files), f"File not found: {file}"  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def check_font(font=FONT, progress=False):
    """
    Checks and downloads the specified font to CONFIG_DIR if it doesn't exist, with an optional download progress
    display.

    Args:
        font (str | Path): Path to the font file, either as a string or Path object. Default is `FONT`.
        progress (bool): If True, show progress of the download. Default is False.

    Returns:
        None

    Note:
        The function ensures the presence of the required font by downloading it from a predefined URL if not available locally.

    Example:
        ```python
        check_font(font="Arial.ttf", progress=True)
        ```
    """
    font = Path(font)
    file = CONFIG_DIR / font.name
    if not font.exists() and not file.exists():
        url = f"https://ultralytics.com/assets/{font.name}"
        LOGGER.info(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, str(file), progress=progress)


def check_dataset(data, autodownload=True):
    """
    Verifies and prepares the dataset by performing optional downloads, checking necessary fields, and unzipping
    archives.

    Args:
        data (str | Path | dict): The dataset source, which can be a path to a YAML file, a URL of a compressed dataset,
            or a dictionary containing dataset details.
        autodownload (bool): If True, attempts to automatically download the dataset if it's not found locally. Default is True.

    Returns:
        dict: A dictionary containing processed dataset configurations, including paths to train and validation data.

    Raises:
        Exception: If the dataset is not found or needed fields are missing from the YAML configuration.

    Example:
        ```python
        from pathlib import Path
        from ultralytics.yolov5.utils.general import check_dataset

        data_config = check_dataset("path/to/dataset.yaml")
        ```

    Notes:
        - Assumes certain fields ('train', 'val', 'names') must be present in the dataset configuration.
        - Converts `names` field to a dictionary if originally provided as a list.
        - If `autodownload` is enabled and the dataset path ends with `.zip`, it downloads and unzips the file.
        - For datasets requiring a specific font, checks and downloads "Arial.ttf" or "Arial.Unicode.ttf" as needed.
    """

    # Download (optional)
    extract_dir = ""
    if isinstance(data, (str, Path)) and (is_zipfile(data) or is_tarfile(data)):
        download(data, dir=f"{DATASETS_DIR}/{Path(data).stem}", unzip=True, delete=False, curl=False, threads=1)
        data = next((DATASETS_DIR / Path(data).stem).rglob("*.yaml"))
        extract_dir, autodownload = data.parent, False

    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        data = yaml_load(data)  # dictionary

    # Checks
    for k in "train", "val", "names":
        assert k in data, emojis(f"data.yaml '{k}:' field missing ❌")
    if isinstance(data["names"], (list, tuple)):  # old array format
        data["names"] = dict(enumerate(data["names"]))  # convert to dict
    assert all(isinstance(k, int) for k in data["names"].keys()), "data.yaml names keys must be integers, i.e. 2: car"
    data["nc"] = len(data["names"])

    # Resolve paths
    path = Path(extract_dir or data.get("path") or "")  # optional 'path' default to '.'
    if not path.is_absolute():
        path = (ROOT / path).resolve()
        data["path"] = path  # download scripts
    for k in "train", "val", "test":
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # Parse yaml
    train, val, test, s = (data.get(x) for x in ("train", "val", "test", "download"))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            LOGGER.info("\nDataset not found ⚠️, missing paths %s" % [str(x) for x in val if not x.exists()])
            if not s or not autodownload:
                raise Exception("Dataset not found ❌")
            t = time.time()
            if s.startswith("http") and s.endswith(".zip"):  # URL
                f = Path(s).name  # filename
                LOGGER.info(f"Downloading {s} to {f}...")
                torch.hub.download_url_to_file(s, f)
                Path(DATASETS_DIR).mkdir(parents=True, exist_ok=True)  # create root
                unzip_file(f, path=DATASETS_DIR)  # unzip
                Path(f).unlink()  # remove zip
                r = None  # success
            elif s.startswith("bash "):  # bash script
                LOGGER.info(f"Running {s} ...")
                r = subprocess.run(s, shell=True)
            else:  # python script
                r = exec(s, {"yaml": data})  # return None
            dt = f"({round(time.time() - t, 1)}s)"
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in (0, None) else f"failure {dt} ❌"
            LOGGER.info(f"Dataset download {s}")
    check_font("Arial.ttf" if is_ascii(data["names"]) else "Arial.Unicode.ttf", progress=True)  # download fonts
    return data  # dictionary


def check_amp(model):
    """
    Checks PyTorch AMP functionality with a model and sample image, ensuring AMP compatibility.

    Args:
        model (torch.nn.Module): The PyTorch model to be tested for AMP compatibility.
    
    Returns:
        bool: True if AMP is supported and functions correctly, False otherwise.
    
    Note:
        AMP is tested only on CUDA devices. If the model is on a CPU or MPS device, AMP support will be automatically deemed
        as unsupported.
    
    Raises:
        Various exceptions could be raised during the AMP compatibility checks, logged as warnings and resulting in AMP
        being marked as incompatible.
        
    Examples:
        ```python
        >>> from models.common import DetectMultiBackend
        >>> import torch
    
        >>> model = DetectMultiBackend('yolov5n.pt')
        >>> torch_amp_supported = check_amp(model)
        >>> print(torch_amp_supported)
        True
        ```
    
    Related Links:
        - [YOLOv5 Issues](https://github.com/ultralytics/yolov5/issues/7908)
    
    ```
    """
    from models.common import AutoShape, DetectMultiBackend

    def amp_allclose(model, im):
        """Compares FP32 and AMP inference results for a model and image, ensuring outputs are within 10% tolerance."""
        m = AutoShape(model, verbose=False)  # model
        a = m(im).xywhn[0]  # FP32 inference
        m.amp = True
        b = m(im).xywhn[0]  # AMP inference
        return a.shape == b.shape and torch.allclose(a, b, atol=0.1)  # close to 10% absolute tolerance

    prefix = colorstr("AMP: ")
    device = next(model.parameters()).device  # get model device
    if device.type in ("cpu", "mps"):
        return False  # AMP only used on CUDA devices
    f = ROOT / "data" / "images" / "bus.jpg"  # image to check
    im = f if f.exists() else "https://ultralytics.com/images/bus.jpg" if check_online() else np.ones((640, 640, 3))
    try:
        assert amp_allclose(deepcopy(model), im) or amp_allclose(DetectMultiBackend("yolov5n.pt", device), im)
        LOGGER.info(f"{prefix}checks passed ✅")
        return True
    except Exception:
        help_url = "https://github.com/ultralytics/yolov5/issues/7908"
        LOGGER.warning(f"{prefix}checks failed ❌, disabling Automatic Mixed Precision. See {help_url}")
        return False


def yaml_load(file="data.yaml"):
    """
    Safely loads a YAML file, ignoring file errors; default file is 'data.yaml'.

    Args:
        file (str | Path): The file path to the YAML file to be loaded. Defaults to "data.yaml".

    Returns:
        dict: The contents of the YAML file as a dictionary.

    Examples:
        ```python
        data_dict = yaml_load("config.yaml")
        ```

    Notes:
        This function uses the SafeLoader from the PyYAML library to load the YAML file. The SafeLoader is used for
        security purposes as it avoids executing arbitrary code inside the YAML file.
    """
    with open(file, errors="ignore") as f:
        return yaml.safe_load(f)


def yaml_save(file="data.yaml", data=None):
    """
    Saves data to a YAML file, converting `Path` objects to strings.

    Args:
        file (str | Path): The file path where the YAML data will be saved. Default is 'data.yaml'.
        data (dict, optional): The data to save into the YAML file. Converts `Path` objects to strings. If None, an empty
            dictionary is used.

    Returns:
        None

    Examples:
        ```python
        from pathlib import Path
        import yaml

        data = {
            'name': 'Ultralytics',
            'version': '1.0',
            'path': Path('/path/to/directory')
        }

        yaml_save('config.yaml', data)
        ```

        This will save the following content to 'config.yaml':
        ```
        name: Ultralytics
        version: '1.0'
        path: /path/to/directory
        ```

    Note:
        Existing content in the file specified by `file` will be overwritten.
    """
    if data is None:
        data = {}
    with open(file, "w") as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


def unzip_file(file, path=None, exclude=(".DS_Store", "__MACOSX")):
    """
    Unzips '*.zip' to `path` (default: file's parent), excluding files matching `exclude` (`('.DS_Store', '__MACOSX')`).

    Args:
        file (str | Path): Path to the zip file.
        path (str | Path, optional): Destination directory where files will be extracted. Defaults to the parent directory of the zip file.
        exclude (tuple, optional): Tuple of substrings. Files containing any of these substrings will be excluded from extraction.

    Returns:
        None

    Examples:
        ```python
        unzip_file('data.zip', '/extracted/path')
        ```

    Notes:
        If `path` is not provided, files are extracted to the parent directory of the zip file.
        Excluded files typically contain metadata or unnecessary system information (e.g., .DS_Store, __MACOSX).
    """
    if path is None:
        path = Path(file).parent  # default path
    with ZipFile(file) as zipObj:
        for f in zipObj.namelist():  # list all archived filenames in the zip
            if all(x not in f for x in exclude):
                zipObj.extract(f, path=path)


def url2file(url):
    """
    Converts a URL to a filename by extracting the last path segment and removing query parameters.

    Args:
        url (str): The URL string from which to extract the filename.

    Returns:
        str: The extracted filename without query parameters.

    Examples:
        ```python
        fname = url2file("https://example.com/path/to/file.txt?param=value")
        print(fname)  # Outputs: 'file.txt'
        ```

    Notes:
        - Ensure the URL is well-formed to avoid unexpected behavior.
        - The function uses pathlib to standardize URL formatting before processing.
    """
    url = str(Path(url)).replace(":/", "://")  # Pathlib turns :// -> :/
    return Path(urllib.parse.unquote(url)).name.split("?")[0]  # '%2F' to '/', split https://url.com/file.txt?auth


def download(url, dir=".", unzip=True, delete=True, curl=False, threads=1, retry=3):
    """
    Downloads files from specified URL(s) into a given directory, optionally unzips, and supports multithreading and
    retries.

    Args:
        url (str | list[str]): A single URL or a list of URLs to be downloaded.
        dir (str | Path, optional): Directory where the files will be saved. Defaults to "." (current directory).
        unzip (bool, optional): If True, unzips the downloaded files if they are in .gz, .zip, or .tar formats. Defaults to True.
        delete (bool, optional): If True, deletes the original downloaded archive file after unzipping. Defaults to True.
        curl (bool, optional): If True, uses curl for downloading. Otherwise, uses torch.hub. Defaults to False.
        threads (int, optional): Number of download threads to use for concurrent downloading. Defaults to 1 (no multithreading).
        retry (int, optional): Number of retry attempts if the download fails. Defaults to 3.

    Returns:
        None

    Examples:
        ```python
        # Download a single file into the current directory and unzip it
        download("https://example.com/file.zip")

        # Download multiple files with multithreading and store in 'data' directory
        download(["https://example.com/file1.zip", "https://example.com/file2.zip"], dir="data", threads=4)

        # Download a file without unzipping or deleting the archive
        download("https://example.com/file.tar.gz", unzip=False, delete=False)
        ```

    Notes:
        - The function attempts to create the specified directory if it does not exist.
        - For many threads, use the optional `curl=True` argument for faster downloads.

    ```
    """

    def download_one(url, dir):
        """Downloads a file from a URL into the specified directory, supporting retries and using curl or torch
        methods.
        """
        success = True
        if os.path.isfile(url):
            f = Path(url)  # filename
        else:  # does not exist
            f = dir / Path(url).name
            LOGGER.info(f"Downloading {url} to {f}...")
            for i in range(retry + 1):
                if curl:
                    success = curl_download(url, f, silent=(threads > 1))
                else:
                    torch.hub.download_url_to_file(url, f, progress=threads == 1)  # torch download
                    success = f.is_file()
                if success:
                    break
                elif i < retry:
                    LOGGER.warning(f"⚠️ Download failure, retrying {i + 1}/{retry} {url}...")
                else:
                    LOGGER.warning(f"❌ Failed to download {url}...")

        if unzip and success and (f.suffix == ".gz" or is_zipfile(f) or is_tarfile(f)):
            LOGGER.info(f"Unzipping {f}...")
            if is_zipfile(f):
                unzip_file(f, dir)  # unzip
            elif is_tarfile(f):
                subprocess.run(["tar", "xf", f, "--directory", f.parent], check=True)  # unzip
            elif f.suffix == ".gz":
                subprocess.run(["tar", "xfz", f, "--directory", f.parent], check=True)  # unzip
            if delete:
                f.unlink()  # remove zip

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multithreaded
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


def make_divisible(x, divisor):
    """
    Adjusts `x` to the nearest value that is greater than or equal to `x` and divisible by `divisor`.

    Args:
        x (int | float): The value to be adjusted to be divisible by `divisor`.
        divisor (int): The value by which `x` needs to be divisible.

    Returns:
        int: The adjusted value of `x` that is divisible by `divisor`.

    Notes:
        If the `divisor` is a Tensor, the maximum value within the Tensor is used as the divisor.

    Examples:
        ```python
        result = make_divisible(65, 8)
        print(result)  # Output: 72
        ```
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def clean_str(s):
    """
    Cleans a string by replacing special characters with underscores.

    Args:
        s (str): Input string to be cleaned.

    Returns:
        str: Cleaned string with special characters replaced by underscores.

    Examples:
        ```python
        clean_str('test@string!')  # returns 'test_string_'
        clean_str('clean-this*string')  # returns 'clean_this_string'
        ```

    Technical Details:
        This function utilizes regular expressions to identify non-alphanumeric characters and replaces them with
        underscores to ensure the resulting string is file-safe and suitable for variable naming or other contexts where
        special characters are not allowed.
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """e_cycle(y1: float = 0.0, y2: float = 1.0, steps: int = 100) -> callable."""
        Generates a function for a sinusoidal ramp from `y1` to `y2` over a specified number of `steps`.
    
        This is commonly used to create learning rate schedules or momentum schedules that follow a single cycle policy.
    
        Args:
            y1 (float, optional): Starting value of the ramp. Defaults to 0.0.
            y2 (float, optional): Ending value of the ramp. Defaults to 1.0.
            steps (int, optional): Number of steps over which the ramp occurs. Defaults to 100.
    
        Returns:
            callable: A function that computes the value at any step `x` within the range [0, steps] based on the sinusoidal ramp.
        
        Example:
            ```python
            # Create a one-cycle function that ramps from 0.1 to 0.9 over 100 steps
            one_cycle_fn = one_cycle(0.1, 0.9, 100)
            
            # Get the value at step 50
            value = one_cycle_fn(50)
            print(value)  # Outputs a value between 0.1 and 0.9 based on the sinusoidal ramp
            ```
    """
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def colorstr(*input):
    """
    Colors strings using ANSI escape codes to enhance readability or emphasize text.
    
    Args:
        *input (str): Variable length string input. The last argument is treated as the string to be colored, 
                      and preceding arguments specify color and style options. 
                      Available colors: "black", "red", "green", "yellow", "blue", 
                      "magenta", "cyan", "white", "bright_black", "bright_red", 
                      "bright_green", "bright_yellow", "bright_blue", "bright_magenta", 
                      "bright_cyan", "bright_white". 
                      Available styles: "bold", "underline".
    
    Returns:
        str: The input string wrapped with ANSI escape codes corresponding to the specified colors and styles.
    
    Examples:
        ```python
        print(colorstr('blue', 'hello world'))  # Blue colored text
        print(colorstr('blue', 'bold', 'hello world'))  # Blue and bold colored text
        ```
    
    Notes:
        - [More on ANSI escape codes](https://en.wikipedia.org/wiki/ANSI_escape_code)
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def labels_to_class_weights(labels, nc=80):
    """
    Calculates class weights for a dataset to counteract class imbalance.
    
    Args:
        labels (list[np.ndarray]): A list of numpy arrays where each array has shape `(n, 5)`, representing `n` label
            instances in the format `[class, x, y, width, height]`.
        nc (int): The number of classes in the dataset. Defaults to 80.
    
    Returns:
        torch.Tensor: A tensor with weights for each class to be used for training, normalized to sum to 1.
    
    Notes:
        - This function is used to provide higher weights to less frequent classes to mitigate data imbalance issues
        during model training.
        
    Examples:
        ```python
        labels = [np.array([[0, 10, 10, 5, 5], [1, 15, 15, 5, 5]]), np.array([[1, 20, 20, 6, 6]])]
        class_weights = labels_to_class_weights(labels, nc=2)
        print(class_weights)
        ```
    """
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights).float()


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    """
    Calculates image weights from labels using class weights, facilitating balanced sampling.
    
    Args:
        labels (list of numpy.ndarray): List of numpy arrays with shape `(n, 5)` where `n` is the number of labels. Each 
            array element pertains to a single image and contains class labels and bounding box information.
        nc (int, optional): Number of classes. Default is `80`.
        class_weights (np.ndarray, optional): Weights for each class, with a default of a ones array of length `80`.
    
    Returns:
        np.ndarray: Array of image weights, where each element corresponds to the weight of a respective image in the dataset.
    
    Example:
        ```python
        labels = [np.array([[0, 0.5, 0.5, 1, 1]]), np.array([[1, 0.5, 0.5, 1, 1]])]
        image_weights = labels_to_image_weights(labels)
        index = random.choices(range(len(image_weights)), weights=image_weights, k=1)  # weighted image sample
        ```
    
    Notes:
        - This function is particularly useful for ensuring that each training batch includes a balanced representation 
          of classes in datasets with class imbalances.
        - It is advisable to perform a sanity check on the labels and class_weights before using this function to ensure 
          they conform to expected shapes and values.
    
    Returns:
        np.ndarray: Array of image weights, one per image in the input 'labels', for balanced sampling.
    """
    # Usage: index = random.choices(range(n), weights=image_weights, k=1)  # weighted image sample
    class_counts = np.array([np.bincount(x[:, 0].astype(int), minlength=nc) for x in labels])
    return (class_weights.reshape(1, nc) * class_counts).sum(1)


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    """
    Converts COCO 80-class indices to COCO 91-class indices.
    
    Returns:
        list: A list of integers mapping COCO 80-class indices to the corresponding COCO 91-class indices.
    
    Reference:
        https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    
    Example:
        ```python
        coco_indices = coco80_to_coco91_class()
        print(coco_indices)  # Prints the mapping from COCO 80 to COCO 91 indices
        ```
    
    Notes:
        - The function generates a predefined static list, representing the mapping from 80-class to 91-class indices.
        - This mapping is based on the alignment of COCO category labels as specified in the provided reference.
    """
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    return [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]


def xyxy2xywh(x):
    """
    Converts bounding boxes from the format `[x1, y1, x2, y2]` to `[x_center, y_center, width, height]`.
    
    Args:
        x (torch.Tensor | np.ndarray): Bounding boxes with shape `(n, 4)` where each box is represented by `[x1, y1, x2, y2]`.
    
    Returns:
        torch.Tensor | np.ndarray: Bounding boxes converted to the format `[x_center, y_center, width, height]` with shape `(n, 4)`.
    
    Examples:
        ```python
        import torch
        x = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])
        y = xyxy2xywh(x)
        print(y)  # should print converted bounding box coordinates
        ```
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    """
    Converts bounding boxes from center format [x, y, w, h] to corner format [x1, y1, x2, y2].
    
    Args:
        x (torch.Tensor | np.ndarray): Input bounding boxes, in the format [x, y, w, h].
    
    Returns:
        torch.Tensor | np.ndarray: Converted bounding boxes, in the format [x1, y1, x2, y2].
    
    Examples:
        ```python
        import torch
        import numpy as np
    
        # Using torch.Tensor
        boxes_tensor = torch.tensor([[50, 50, 20, 20], [30, 30, 10, 10]])
        converted_tensor = xywh2xyxy(boxes_tensor)
        print(converted_tensor)
        # tensor([[40., 40., 60., 60.],
        #         [25., 25., 35., 35.]])
    
        # Using np.ndarray
        boxes_array = np.array([[50, 50, 20, 20], [30, 30, 10, 10]])
        converted_array = xywh2xyxy(boxes_array)
        print(converted_array)
        # array([[40., 40., 60., 60.],
        #        [25., 25., 35., 35.]])
        ```
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """
    Converts bounding boxes from normalized [x, y, w, h] to [x1, y1, x2, y2] format, applying optional padding.
    
    Args:
        x (torch.Tensor | np.ndarray): The input bounding boxes in normalized format [x, y, w, h].
        w (int): Width of the image the bounding box is normalized against (default is 640).
        h (int): Height of the image the bounding box is normalized against (default is 640).
        padw (int): Optional padding to apply on the x-axis; defaults to 0.
        padh (int): Optional padding to apply on the y-axis; defaults to 0.
    
    Returns:
        torch.Tensor | np.ndarray: Bounding boxes in [x1, y1, x2, y2] format with applied padding.
    
    Example:
        ```python
        import torch
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        converted_boxes = xywhn2xyxy(boxes, w=640, h=640)
        print(converted_boxes)  # tensor([[256.0, 256.0, 384.0, 384.0]])
        ```
        
    Notes:
        The function supports both PyTorch tensors and NumPy arrays for input bounding boxes.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """
    Converts bounding boxes from the `[x1, y1, x2, y2]` format to the normalized `[x, y, w, h]` format.
    
    Args:
        x (torch.Tensor | np.ndarray): Input bounding boxes in `[x1, y1, x2, y2]` format.
        w (int, optional): Width of the image. Defaults to 640.
        h (int, optional): Height of the image. Defaults to 640.
        clip (bool, optional): If True, clips the bounding boxes to be within the image boundaries. Defaults to False.
        eps (float, optional): Small epsilon value to adjust boundaries when clipping. Defaults to 0.0.
    
    Returns:
        (torch.Tensor | np.ndarray): Bounding boxes converted to the normalized `[x, y, w, h]` format.
    
    Notes:
        This function supports both PyTorch tensors and NumPy arrays as input.
    
    Example:
        ```python
        import torch
    
        # Example bounding boxes in [x1, y1, x2, y2] format
        boxes = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.float32)
        
        # Convert to normalized [x, y, w, h] format
        normalized_boxes = xyxy2xywhn(boxes, w=640, h=480)
        print(normalized_boxes)
        ```
    
        Given `clip=True`, the function will clip bounding boxes within the image boundaries:
    
        ```python
        clipped_boxes = xyxy2xywhn(boxes, w=640, h=480, clip=True)
        print(clipped_boxes)
        ```
    """
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    """
    Converts normalized coordinates to pixel coordinates, adjusting for width, height, and padding.
    
    Args:
      x (torch.Tensor | np.ndarray): Normalized coordinates of shape (n, 2).
      w (int): Image width. Default is 640.
      h (int): Image height. Default is 640.
      padw (int): Width padding applied to the normalized coordinates. Default is 0.
      padh (int): Height padding applied to the normalized coordinates. Default is 0.
    
    Returns:
      torch.Tensor | np.ndarray: Pixel coordinates of shape (n, 2).
    
    Examples:
      ```
      import torch
      normalized_coords = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
      pixel_coords = xyn2xy(normalized_coords, w=800, h=600)
      print(pixel_coords)
      ```
    References:
      Ultralytics Repository: https://github.com/ultralytics/ultralytics
    
    Note:
      Ensure `x` is either a PyTorch tensor or a NumPy array for proper execution.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * x[..., 0] + padw  # top left x
    y[..., 1] = h * x[..., 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    """
    Segment to bounding box conversion using image dimensions.
    
    Args:
        segment (np.ndarray): Array of shape (n, 2) representing the segment with coordinates in the format [x, y].
        width (int): Width of the image in pixels. Default is 640.
        height (int): Height of the image in pixels. Default is 640.
    
    Returns:
        np.ndarray: Array of shape (4,) representing the bounding box [x1, y1, x2, y2], clipped to stay within image boundaries.
        
    Notes:
        This function ensures that the resulting bounding box coordinates do not exceed the image dimensions.
        
    Example:
        ```python
        segment = np.array([[100, 150], [200, 250], [300, 350]])
        box = segment2box(segment, width=640, height=640)
        print(box)  # Output: [100, 150, 300, 350]
        ```
    """
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    (
        x,
        y,
    ) = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    """
    Converts a list of segmentation coordinates to bounding box coordinates.
    
    Args:
        segments (list[np.ndarray]): A list of segmentation arrays. Each array consists of Nx2 coordinates representing
                                     the vertices of the segmentation polygon.
    
    Returns:
        np.ndarray: An array of bounding boxes in xyxy format, where each bounding box is represented as [x_min, y_min,
                    x_max, y_max].
    
    Notes:
        Segmentation data should be provided as a list where each entry is a numpy array containing the vertices of the
        polygon. The function automatically computes the minimal enclosing box for each segmentation polygon. If no segments
        are provided, the function will return an empty array.
    
    Example:
        ```python
        segments = [np.array([[10, 20], [30, 40], [50, 60]])]
        boxes = segments2boxes(segments)
        print(boxes)  # Output: [[10, 20, 50, 60]]
        ```
    
    Returns:
        np.ndarray: An array of bounding boxes in the format (cls, x_min, y_min, x_max, y_max).
         
    链接：
    *原始链接*
    """
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    """
    Resamples segments with a fixed number of points.
    
    Args:
        segments (list[np.ndarray]): List of segment arrays, each of shape (m, 2), where m is the number of points in
            the segment, and each point is represented by its (x, y) coordinates.
        n (int): Number of points to resample to. Default is 1000.
    
    Returns:
        list[np.ndarray]: List of resampled segment arrays, each of shape (n, 2), where n is the number of resampled
            points, and each point is represented by its (x, y) coordinates.
    
    Example:
        ```python
        segments = [np.array([[0, 0], [1, 1]])]
        resampled_segments = resample_segments(segments, n=5)
        print(resampled_segments)
        # Output: [array([[0. , 0. ], [0.25, 0.25], [0.5 , 0.5 ], [0.75, 0.75], [1. , 1. ]])]
        ```
    
    Note:
        The function ensures that the segments are closed by appending the first point to the end before resampling. This
        is typically required for closed polygon operations.
    """
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    Scales bounding boxes from one image shape to another with optional ratio and padding adjustments.
    
    Args:
        img1_shape (tuple[int, int]): Shape of the new image as (height, width).
        boxes (np.ndarray | torch.Tensor): Array of bounding boxes in the format [x1, y1, x2, y2]. Shape is (n, 4) 
                                            where n is the number of boxes.
        img0_shape (tuple[int, int]): Shape of the original image as (height, width).
        ratio_pad (Optional[tuple[tuple[float, float], tuple[float, float]]]): Tuple containing scaling ratio and padding 
                                                                               as ((gain_x, gain_y), (pad_x, pad_y)). 
                                                                               Default is None.
    
    Returns:
        None: The boxes are modified in place to fit the new image shape.
    
    Example:
        ```python
        img1_shape = (640, 480)
        boxes = np.array([[50, 50, 150, 150], [30, 30, 60, 60]])
        img0_shape = (320, 240)
        scale_boxes(img1_shape, boxes, img0_shape)
        ```
    Note:
        This function directly modifies the input `boxes` array.
        Ensure to use the function within the correct context of image preprocessing or post-processing.
    
    For more details, refer to: https://github.com/ultralytics/ultralytics
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def scale_segments(img1_shape, segments, img0_shape, ratio_pad=None, normalize=False):
    """
    Rescales segment coordinates from img1_shape to img0_shape, optionally normalizing, and supports padding adjustments.
    
    Args:
        img1_shape (tuple): Shape of the image from which the segments originate, formatted as (height, width).
        segments (numpy.ndarray): Segment coordinates to be transformed, shaped as (n, 2), where n is the number of points.
        img0_shape (tuple): Shape of the target image to which the segments are rescaled, formatted as (height, width).
        ratio_pad (Optional[tuple]): Ratio and padding information, formatted as ((ratio_h, ratio_w), (pad_w, pad_h)). Defaults to None.
        normalize (bool, optional): If True, normalizes the coordinates to the range [0, 1]. Defaults to False.
    
    Returns:
        numpy.ndarray: Rescaled segment coordinates, potentially normalized, shaped as (n, 2).
    
    Notes:
        This function adjusts segment coordinates based on the scaling factor derived from the size difference between
        img1_shape and img0_shape. It handles any padding that might have been applied and can normalize the output if
        requested. The resulting coordinates are clipped to ensure they stay within the image boundaries.
    
    Example:
        ```python
        img1_shape = (720, 1280)
        segments = np.array([[100, 200], [150, 250], [200, 300]])
        img0_shape = (640, 640)
        scaled_segments = scale_segments(img1_shape, segments, img0_shape, normalize=True)
        ```
    
    See also:
        - utils.general.clip_segments: Function to clip the coordinates of segment points.
        - utils.plots.plot_segments: Function to visualize segments on an image.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    segments[:, 0] -= pad[0]  # x padding
    segments[:, 1] -= pad[1]  # y padding
    segments /= gain
    clip_segments(segments, img0_shape)
    if normalize:
        segments[:, 0] /= img0_shape[1]  # width
        segments[:, 1] /= img0_shape[0]  # height
    return segments


def clip_boxes(boxes, shape):
    """
    Clips bounding box coordinates to lie within image boundaries.
    
    Args:
        boxes (torch.Tensor | np.ndarray): Bounding boxes in [x1, y1, x2, y2] format. Supports both torch tensors and numpy arrays.
        shape (tuple[int, int]): Shape of the image as (height, width).
    
    Returns:
        None
    
    Notes:
        This function modifies the `boxes` array in place to ensure all coordinates lie within the image shape.
        For `torch.Tensor`, individual coordinates are clamped for potentially faster operation, while for `numpy.ndarray`,
        coordinates are clipped in groups for efficiency.
    
    Example:
        ```python
        import torch
        import numpy as np
    
        boxes = torch.tensor([[10, 20, 300, 400], [-10, -20, 500, 600]])
        image_shape = (480, 640)
        clip_boxes(boxes, image_shape)
        print(boxes)
    
        boxes_np = np.array([[10, 20, 300, 400], [-10, -20, 500, 600]])
        clip_boxes(boxes_np, image_shape)
        print(boxes_np)
        ```
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def clip_segments(segments, shape):
    """
    Clip segments within the specified image dimensions, keeping them within the boundary limits.
    
    Args:
        segments (np.array | torch.Tensor): The array of segments where each segment is described by a set of coordinates.
        shape (tuple): The shape of the image as (height, width) within which segments need to be clipped.
    
    Returns:
        None: This function modifies the `segments` array in-place.
    
    Notes:
        - This function supports both NumPy arrays and PyTorch tensors as input for `segments`.
        - Warning: Clipping is performed in-place, altering the input `segments`.
    
    Examples:
        ```python
        import numpy as np
    
        # Example with NumPy array
        segments = np.array([[0, 10], [50, 100], [80, 200]])
        shape = (100, 100)
        clip_segments(segments, shape)
        # segments is now clipped within the bounds of shape
    
        import torch
    
        # Example with torch.Tensor
        segments = torch.Tensor([[0, 10], [50, 100], [80, 200]])
        shape = (100, 100)
        clip_segments(segments, shape)
        # segments is now clipped within the bounds of shape
        ```
    """
    if isinstance(segments, torch.Tensor):  # faster individually
        segments[:, 0].clamp_(0, shape[1])  # x
        segments[:, 1].clamp_(0, shape[0])  # y
    else:  # np.array (faster grouped)
        segments[:, 0] = segments[:, 0].clip(0, shape[1])  # x
        segments[:, 1] = segments[:, 1].clip(0, shape[0])  # y


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.
    
    Args:
        prediction (torch.Tensor): Inference results of shape (N, num_classes + 5 + num_masks) from the model.
        conf_thres (float): Confidence threshold for filtering initial boxes. Default is 0.25.
        iou_thres (float): Intersection-over-union (IoU) threshold for NMS. Default is 0.45.
        classes (list[int], optional): List of class indices to keep. If `None`, all classes are considered. Default is `None`.
        agnostic (bool): If True, class-agnostic NMS is applied. Default is False.
        multi_label (bool): If True, boxes can belong to multiple classes. Default is False.
        labels (list[torch.Tensor]): List of labeled boxes to include, one per image. Default is an empty tuple.
        max_det (int): Maximum number of detections to keep after NMS per image. Default is 300.
        nm (int): Number of mask columns in the predicted tensor, if any. Default is 0.
    
    Returns:
        list[torch.Tensor]: List of tensors, each of shape (N, 6+nm), representing detections for each image. Each tensor
                            has columns [x1, y1, x2, y2, conf, cls, mask0, mask1, ...]. The number of mask columns equals nm.
    
    Raises:
        AssertionError: If `conf_thres` or `iou_thres` are not within the range [0, 1].
    
    Example:
        ```python
        detections = non_max_suppression(prediction, conf_thres=0.5, iou_thres=0.5)
        ```
    
    Notes:
        - NMS filters out overlapping bounding boxes based on their confidence scores and IoU values.
        - Supports detection with masks if the model's output contains masks.
        - Hyperparameters like `conf_thres` and `iou_thres` can greatly impact the resulting detections.
    """

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv3 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output


def strip_optimizer(f="best.pt", s=""):  # from utils.general import *; strip_optimizer()
    """
    strip_optimizer(f: str = "best.pt", s: str = "") -> None
        """
        Removes the optimizer state and auxiliary components from a model checkpoint, reducing its size for inference.

        Args:
            f (str): Path to the checkpoint file to be stripped. Default is "best.pt".
            s (str): Path to save the stripped checkpoint file. If empty, it will overwrite the original file.

        Returns:
            None

        Notes:
            The function will load the checkpoint from the file `f`, remove the optimizer state, best fitness, EMA, and updates.
            It will then convert the model weights to half precision (FP16) and set requires_grad to False for all parameters
            to optimize storage. The stripped checkpoint is then saved to either the original file `f` or to a new file `s`
            if provided.

        Example:
            ```python
            from ultralytics import utils

            utils.strip_optimizer("path/to/checkpoint.pt", "path/to/stripped_checkpoint.pt")
            ```
        """
    x = torch.load(f, map_location=torch.device("cpu"))
    if x.get("ema"):
        x["model"] = x["ema"]  # replace model with ema
    for k in "optimizer", "best_fitness", "ema", "updates":  # keys
        x[k] = None
    x["epoch"] = -1
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1e6  # filesize
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")


def print_mutation(keys, results, hyp, save_dir, bucket, prefix=colorstr("evolve: ")):
    """
    Logs mutation results, updates evolve CSV/YAML, optionally syncs with cloud storage.

    Args:
        keys (list[str]): List of metric and hyperparameter names.
        results (tuple): Tuple of metric values.
        hyp (dict): Dictionary of hyperparameters and their values.
        save_dir (Path): Directory to save the evolve logs.
        bucket (str): Name of the cloud storage bucket for syncing.
        prefix (str, optional): Prefix for log messages.

    Returns:
        None

    Examples:
        ```python
        keys = ['metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss', 'val/obj_loss', 'val/cls_loss']
        results = (0.5, 0.4, 0.1, 0.05, 0.01)
        hyp = {'lr0': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 'giou': 0.05}
        save_dir = Path('/path/to/save_dir')
        bucket = 'my-bucket-name'

        print_mutation(keys, results, hyp, save_dir, bucket)
        ```
    Notes:
        - This function is used within hyperparameter evolution to log and analyze the results of each mutation.
        - Make sure the required dependencies like pandas, numpy, and gsutil are available in your environment.
    """
    evolve_csv = save_dir / "evolve.csv"
    evolve_yaml = save_dir / "hyp_evolve.yaml"
    keys = tuple(keys) + tuple(hyp.keys())  # [results + hyps]
    keys = tuple(x.strip() for x in keys)
    vals = results + tuple(hyp.values())
    n = len(keys)

    # Download (optional)
    if bucket:
        url = f"gs://{bucket}/evolve.csv"
        if gsutil_getsize(url) > (evolve_csv.stat().st_size if evolve_csv.exists() else 0):
            subprocess.run(["gsutil", "cp", f"{url}", f"{save_dir}"])  # download evolve.csv if larger than local

    # Log to evolve.csv
    s = "" if evolve_csv.exists() else (("%20s," * n % keys).rstrip(",") + "\n")  # add header
    with open(evolve_csv, "a") as f:
        f.write(s + ("%20.5g," * n % vals).rstrip(",") + "\n")

    # Save yaml
    with open(evolve_yaml, "w") as f:
        data = pd.read_csv(evolve_csv, skipinitialspace=True)
        data = data.rename(columns=lambda x: x.strip())  # strip keys
        i = np.argmax(fitness(data.values[:, :4]))  #
        generations = len(data)
        f.write(
            "# YOLOv3 Hyperparameter Evolution Results\n"
            + f"# Best generation: {i}\n"
            + f"# Last generation: {generations - 1}\n"
            + "# "
            + ", ".join(f"{x.strip():>20s}" for x in keys[:7])
            + "\n"
            + "# "
            + ", ".join(f"{x:>20.5g}" for x in data.values[i, :7])
            + "\n\n"
        )
        yaml.safe_dump(data.loc[i][7:].to_dict(), f, sort_keys=False)

    # Print to screen
    LOGGER.info(
        prefix
        + f"{generations} generations finished, current result:\n"
        + prefix
        + ", ".join(f"{x.strip():>20s}" for x in keys)
        + "\n"
        + prefix
        + ", ".join(f"{x:20.5g}" for x in vals)
        + "\n\n"
    )

    if bucket:
        subprocess.run(["gsutil", "cp", f"{evolve_csv}", f"{evolve_yaml}", f"gs://{bucket}"])  # upload


def apply_classifier(x, model, img, im0):
    """
    Apply a secondary classifier to YOLO output detections, adjusting bounding boxes to square shapes, rescaling, and
    filtering matches.

    Args:
        x (torch.Tensor): Tensor containing YOLO detection outputs.
        model (torchvision.models): Pre-trained classifier model.
        img (torch.Tensor): YOLO input image tensor.
        im0 (np.ndarray | list[np.ndarray]): Original images, either a single image (ndarray) or a list of images (list of ndarray).

    Returns:
        None: The function updates `x` in place by filtering detections with the classifier.

    Examples:
        ```python
        import torchvision.models as models
        import torch
        import cv2
        import numpy as np

        # Example secondary classification model
        classifier = models.efficientnet_b0(pretrained=True).to(device).eval()

        # Example use case
        detections = ...  # YOLO detections tensor
        img_tensor = ...  # YOLO input image tensor
        original_images = ...  # List of original images (numpy arrays)

        apply_classifier(detections, classifier, img_tensor, original_images)
        ```

    Note:
        Rescaling and reshaping bounding boxes ensure compatibility between YOLO detections and the secondary classifier.
        This function assumes `img` and `im0` are preprocessed correctly for YOLO and classifier models, respectively.
    """
    # Example model = torchvision.models.__dict__['efficientnet_b0'](pretrained=True).to(device).eval()
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_boxes(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for a in d:
                cutout = im0[i][int(a[1]) : int(a[3]), int(a[0]) : int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Increments a file or directory path by appending a number to make a unique path if it already exists.

    Args:
        path (str | Path): The base path to increment.
        exist_ok (bool): If True, the function won't increment the path if it already exists. Defaults to False.
        sep (str): Separator to use between the original path and the incremented number. Defaults to an empty string.
        mkdir (bool): If True, creates the directory (if a directory path is given) using the newly incremented path. Defaults to False.

    Returns:
        Path: The incremented file or directory path.

    Examples:
        To get a unique directory name:
        ```python
        p = increment_path('runs/exp', mkdir=True)
        print(p)  # Outputs 'runs/exp2' if 'runs/exp' already exists
        ```

    Notes:
        This function uses a limit of 9998 as the maximum number for increments. Beyond this, it will not search for unique paths. Ensure this limit suffices for your use case.
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


# OpenCV Multilanguage-friendly functions
# ------------------------------------------------------------------------------------
imshow_ = cv2.imshow  # copy to avoid recursion errors


def imread(filename, flags=cv2.IMREAD_COLOR):
    """
    Reads an image from a file, supporting multilanguage paths, and returns it in the specified color scheme.

    Args:
        filename (str | Path): Path to the image file.
        flags (int): Flag that specifies the color type of the loaded image. Defaults to `cv2.IMREAD_COLOR`.

    Returns:
        img (ndarray | None): The loaded image, or `None` if unable to load.

    Examples:
        ```python
        import cv2
        from ultralytics.utils.general import imread

        img = imread('path/to/image.jpg', cv2.IMREAD_GRAYSCALE)
        if img is not None:
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error loading image.")
        ```
    """
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)


def imwrite(filename, img):
    """
    Writes an image to a specified file path, supporting multilanguage paths.

    Args:
        filename (str | Path): The path to the file where the image will be saved. Supports Path objects for better path handling.
        img (np.ndarray): The image to be saved. It should be a valid NumPy array representation of the image.

    Returns:
        bool: True if the image was successfully written, False otherwise.

    Examples:
        To save an image:

        ```python
        import cv2
        import numpy as np
        from ultralytics.utils.general import imwrite

        # Creating a dummy image
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Save the image
        success = imwrite('path/to/image.png', img)
        print("Image saved:", success)
        ```
    """
    try:
        cv2.imencode(Path(filename).suffix, img)[1].tofile(filename)
        return True
    except Exception:
        return False


def imshow(path, im):
    """
    Displays an image using OpenCV, handling path as title and image data as content.

    Args:
        path (str): Title of the window where the image will be displayed.
        im (ndarray): Image data array to be displayed, in the format (height, width, channels).

    Returns:
        None

    Notes:
        - This method uses OpenCV's `cv2.imshow` function to render the image in a window.
        - Ensure the display environment supports OpenCV GUI functions; otherwise, this function might fail.
        - For environments like notebooks or servers where GUI display is not possible, alternative visualization methods might be needed.

    Example:
        ```python
        import cv2
        from ultralytics.utils import imshow

        img = cv2.imread('path/to/image.jpg')
        imshow('Image Title', img)
        cv2.waitKey(0)  # Waits for a key press to close the window
        cv2.destroyAllWindows()  # Closes the window
        ```
    """
    imshow_(path.encode("unicode_escape").decode(), im)


if Path(inspect.stack()[0].filename).parent.parent.as_posix() in inspect.stack()[-1].filename:
    cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow  # redefine

# Variables ------------------------------------------------------------------------------------------------------------
