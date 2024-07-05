# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""Download utils."""

import logging
import subprocess
import urllib
from pathlib import Path

import requests
import torch


def is_url(url, check=True):
    """
    Determines if a string is a valid URL and optionally checks its existence online.

    Args:
        url (str | pathlib.Path): The string or Path object to be evaluated as a URL.
        check (bool): If True, the function will check whether the URL exists online. Defaults to True.

    Returns:
        bool: True if the input is a valid URL (and exists online if `check` is True), otherwise False.

    Raises:
        None: This function is designed to catch its own exceptions internally.

    Examples:
        ```python
        url = "https://ultralytics.com"
        is_valid = is_url(url)  # Returns True if the URL is valid and reachable
        ```
    """
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        return (urllib.request.urlopen(url).getcode() == 200) if check else True  # check if exists online
    except (AssertionError, urllib.request.HTTPError):
        return False


def gsutil_getsize(url=""):
    """
    Retrieves the size of a file located at a Google Cloud Storage URL using the `gsutil du` command.

    Args:
        url (str): The Google Cloud Storage URL (e.g., 'gs://bucket_name/object_name').

    Returns:
        int: Size of the file in bytes. Returns 0 if the file is not found or if the `gsutil du` command fails.

    Notes:
        This function assumes that the `gsutil` command-line tool is installed and properly configured on the local machine.

    Examples:
        ```python
        file_size = gsutil_getsize('gs://my_bucket/my_file.txt')
        print(file_size)  # Outputs the size of the file in bytes or 0 if not found
        ```
    """
    output = subprocess.check_output(["gsutil", "du", url], shell=True, encoding="utf-8")
    return int(output.split()[0]) if output else 0


def url_getsize(url="https://ultralytics.com/images/bus.jpg"):
    """
    Fetches the file size in bytes from a given URL using an HTTP HEAD request.

    Args:
        url (str): The URL of the file for which to get the size. Defaults to 'https://ultralytics.com/images/bus.jpg'.

    Returns:
        int: The size of the file in bytes if the request is successful; otherwise, -1.

    Examples:
        ```python
        size = url_getsize('https://ultralytics.com/images/bus.jpg')
        print(size)  # Example output: 23456
        ```

    Notes:
        This function sends an HTTP HEAD request to the specified URL to determine the size of the file. If the request
        fails or the file size cannot be determined, the function returns -1. The `requests` library is used for handling
        the HTTP request.
    """
    response = requests.head(url, allow_redirects=True)
    return int(response.headers.get("content-length", -1))


def curl_download(url, filename, *, silent: bool = False) -> bool:
    """
    Download a file from a specified URL to a given filename using the curl command line tool.

    Args:
        url (str): The URL from which to download the file.
        filename (str | Path): The local path where the file will be saved.
        silent (bool, optional): If True, suppress curl's progress meter and error messages. Defaults to False.

    Returns:
        bool: True if the download succeeded (exit code 0), otherwise False.

    Examples:
        ```python
        url = "https://ultralytics.com/images/bus.jpg"
        filename = "bus.jpg"
        success = curl_download(url, filename, silent=True)
        print("Download successful:", success)
        ```

    Notes:
        - curl command line tool must be installed and accessible in the system PATH.
        - The function uses curl's built-in resume functionality and retry logic to handle intermittent network issues.
    """
    silent_option = "sS" if silent else ""  # silent
    proc = subprocess.run(
        [
            "curl",
            "-#",
            f"-{silent_option}L",
            url,
            "--output",
            filename,
            "--retry",
            "9",
            "-C",
            "-",
        ]
    )
    return proc.returncode == 0


def safe_download(file, url, url2=None, min_bytes=1e0, error_msg=""):
    """
    Downloads a file from a given URL or secondary URL to a specified file path, ensuring the file size exceeds a
    minimum threshold; removes incomplete downloads if conditions are not met.

    Args:
        file (str | Path): The file destination where the downloaded content will be saved.
        url (str): The primary URL to download the file from.
        url2 (str | None): The secondary URL to download the file from if the primary URL fails. Defaults to None.
        min_bytes (int | float): The minimum file size in bytes. Ensures the downloaded file meets this size
            requirement. Defaults to 1.
        error_msg (str): An additional error message to log if the download fails. Defaults to an empty string.

    Returns:
        None

    Notes:
        - Utilizes `torch.hub.download_url_to_file` for downloading and provides re-attempt logic using an alternative URL
          if available.
        - Checks if the downloaded file exists and its size is greater than `min_bytes`. If checks fail, deletes the incomplete
          download and logs an error message.
    """
    from utils.general import LOGGER

    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        LOGGER.info(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, str(file), progress=LOGGER.level <= logging.INFO)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        if file.exists():
            file.unlink()  # remove partial downloads
        LOGGER.info(f"ERROR: {e}\nRe-attempting {url2 or url} to {file}...")
        # curl download, retry and resume on fail
        curl_download(url2 or url, file)
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            if file.exists():
                file.unlink()  # remove partial downloads
            LOGGER.info(f"ERROR: {assert_msg}\n{error_msg}")
        LOGGER.info("")


def attempt_download(file, repo="ultralytics/yolov5", release="v7.0"):
    """
    Attempts to download a file from a specified URL or GitHub release, ensuring file integrity with a minimum size
    check.

    Args:
        file (str | Path): The target file path to download to.
        repo (str): The GitHub repository to download from. Defaults to 'ultralytics/yolov5'.
        release (str): The release version to download. Defaults to 'v7.0'.

    Returns:
        Path: The path to the downloaded file.

    Notes:
        - Ensures that the downloaded file is complete by checking its size.
        - Can handle downloads from direct URLs or GitHub assets.
        - Utilizes helper functions for safe downloading and size checking.

    Examples:
        ```python
        from ultralytics import attempt_download

        # Attempt to download a file directly from a URL
        downloaded_file = attempt_download('https://example.com/path/to/file.zip')

        # Attempt to download a GitHub release asset
        downloaded_file = attempt_download('yolov5s.pt')
        ```
    """
    from utils.general import LOGGER

    def github_assets(repository, version="latest"):
        """Returns GitHub tag and assets for a given repository and version from the GitHub API."""
        if version != "latest":
            version = f"tags/{version}"  # i.e. tags/v7.0
        response = requests.get(f"https://api.github.com/repos/{repository}/releases/{version}").json()  # github api
        return response["tag_name"], [x["name"] for x in response["assets"]]  # tag, assets

    file = Path(str(file).strip().replace("'", ""))
    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(("http:/", "https:/")):  # download
            url = str(file).replace(":/", "://")  # Pathlib turns :// -> :/
            file = name.split("?")[0]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f"Found {url} locally at {file}")  # file already exists
            else:
                safe_download(file=file, url=url, min_bytes=1e5)
            return file

        # GitHub assets
        assets = [f"yolov5{size}{suffix}.pt" for size in "nsmlx" for suffix in ("", "6", "-cls", "-seg")]  # default
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)  # latest release
            except Exception:
                try:
                    tag = subprocess.check_output("git tag", shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
                except Exception:
                    tag = release

        if name in assets:
            file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
            safe_download(
                file,
                url=f"https://github.com/{repo}/releases/download/{tag}/{name}",
                min_bytes=1e5,
                error_msg=f"{file} missing, try downloading from https://github.com/{repo}/releases/{tag}",
            )

    return str(file)
