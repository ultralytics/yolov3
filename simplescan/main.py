#!/usr/bin/env python3
import argparse
import asyncio
import concurrent.futures
import configparser
import faulthandler
import json
import logging
import os
import pathlib
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from timeit import default_timer as timer
from typing import Any, Dict

import paho.mqtt.client as paho
import requests
import sdnotify

from camera import Camera
from detect import detect
from homeassistant import HomeAssistant
from object_detection_rtv4 import ONNXTensorRTv4ObjectDetection
from utils import cleanup

log: logging.Logger = logging.getLogger("aicam")
mlog: logging.Logger = logging.getLogger("mqtt")
kill_now: bool = False

DEVICE_ID = "aicam"
DEVICE_NAME = "AI Camera Detector"


def get_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "describe", "--always", "--dirty"],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def device_info(version: str) -> dict:
    return {
        "identifiers": [DEVICE_ID],
        "name": DEVICE_NAME,
        "manufacturer": "aicam",
        "model": "Jetson Nano Detector",
        "sw_version": version,
    }


def on_publish(client: paho.Client, userdata: Any, mid: int) -> None:
    mlog.debug("on_publish({},{})".format(userdata, mid))


def on_connect(client: paho.Client, userdata: Any, flags: Dict, rc: int) -> None:
    mlog.info("mqtt connected")
    client._reconnect_deadline = None
    client.publish("aicam/status", "online", retain=True)


def on_disconnect(client: paho.Client, userdata: Any, rc: int) -> None:
    if rc == 0:
        mlog.info("mqtt disconnected cleanly")
        return
    mlog.warning("mqtt disconnected unexpectedly, reason=%s. Will retry for 5 minutes.", rc)
    client._reconnect_deadline = time.monotonic() + 300


def on_message(
    self: Any, mqtt_client: paho.Client, obj: Any, msg: paho.MQTTMessage
) -> None:
    mlog.info("on_message()")


class GracefulKiller:
    def __init__(self) -> None:
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args: Any) -> None:
        global kill_now
        kill_now = True


async def main(options: argparse.Namespace) -> None:
    config: configparser.ConfigParser = configparser.ConfigParser()
    config.read(options.config_file)
    ha: HomeAssistant = HomeAssistant(config["homeassistant"])
    detector_config: Dict[str, str] = config["detector"]
    color_model_config: Dict[str, str] = config["color-model"]
    grey_model_config: Dict[str, str] = config["grey-model"]
    mqtt_icons: Dict[str, str] = config["mqtt_icons"]
    lwt: str = "aicam/status"
    mqtt_client: paho.Client = paho.Client(client_id="aicam")
    mqtt_client.enable_logger(logger=mlog)
    mqtt_client.on_publish = on_publish
    mqtt_client.on_connect = on_connect
    mqtt_client.on_disconnect = on_disconnect
    mqtt_client.on_message = on_message
    mqtt_client.will_set(lwt, payload="offline", qos=0, retain=True)
    mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)
    mqtt_client._reconnect_deadline = None
    mqtt_config = config["mqtt"]
    mqtt_client.username_pw_set(mqtt_config["user"], mqtt_config["password"])
    try:
        mqtt_client.connect(mqtt_config["host"], mqtt_config.getint("port", 1883), keepalive=60)
    except Exception as e:
        log.error(f"Failed to connect to MQTT broker: {e}")
        raise
    mqtt_client.subscribe("test")  # get on connect messages
    mqtt_client.loop_start()

    # Load labels
    with open(detector_config["labelfile-path"], "r") as f:
        labels = [line.strip() for line in f.readlines()]
    vehicle_labels = []
    if "vehicle-labelfile-path" in detector_config:
        with open(detector_config["vehicle-labelfile-path"], "r") as f:
            vehicle_labels = [line.strip() for line in f.readlines()]
    # open static exclusion
    excludes = {}
    if "excludes-file" in detector_config:
        with open(detector_config["excludes-file"]) as f:
            excludes = json.load(f)
    # make dirs
    static_dir = os.path.join(config["detector"]["save-path"], "static")
    pathlib.Path(static_dir).mkdir(parents=True, exist_ok=True)

    sd = sdnotify.SystemdNotifier()
    sd.notify("STATUS=Loading color model")
    color_model = ONNXTensorRTv4ObjectDetection(color_model_config, labels)
    sd.notify("STATUS=Loading grey model")
    grey_model = ONNXTensorRTv4ObjectDetection(grey_model_config, labels)
    sd.notify("STATUS=Loading vehicle/packages model")
    vehicle_model = ONNXTensorRTv4ObjectDetection(
        config["vehicle-model"], vehicle_labels
    )
    sd.notify("STATUS=Loaded models")

    cams = []
    i = 0
    while "cam%d" % i in config.sections():
        cams.append(
            Camera(config["cam%d" % i], excludes.get(config["cam%d" % i]["name"], {}), mqtt_config)
        )
        i += 1
    log.info("Configured %i cams" % i)
    async_cameras = len(list(filter(lambda cam: cam.capture_async, cams)))
    # async_cameras = 4
    log.info("%i async workers" % async_cameras)
    if async_cameras > 0 and not options.sync:
        async_pool = concurrent.futures.ThreadPoolExecutor(max_workers=async_cameras)
    else:
        async_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    version = get_version()
    dev = device_info(version)

    # Publish device-level diagnostic sensors
    mqtt_client.publish(
        f"homeassistant/sensor/{DEVICE_ID}-version/config",
        json.dumps(
            {
                "name": "Version",
                "state_topic": f"{DEVICE_ID}/version",
                "uniq_id": f"{DEVICE_ID}-version",
                "availability_topic": lwt,
                "icon": "mdi:tag",
                "entity_category": "diagnostic",
                "device": dev,
            }
        ),
        retain=True,
    )
    mqtt_client.publish(f"{DEVICE_ID}/version", version, retain=True)

    mqtt_client.publish(
        f"homeassistant/sensor/{DEVICE_ID}-status/config",
        json.dumps(
            {
                "name": "Status",
                "state_topic": f"{DEVICE_ID}/device_status",
                "uniq_id": f"{DEVICE_ID}-status",
                "availability_topic": lwt,
                "icon": "mdi:information-outline",
                "entity_category": "diagnostic",
                "device": dev,
            }
        ),
        retain=True,
    )
    mqtt_client.publish(f"{DEVICE_ID}/device_status", "running", retain=True)

    mqtt_client.publish(
        f"homeassistant/sensor/{DEVICE_ID}-cameras/config",
        json.dumps(
            {
                "name": "Camera Count",
                "state_topic": f"{DEVICE_ID}/camera_count",
                "uniq_id": f"{DEVICE_ID}-cameras",
                "availability_topic": lwt,
                "icon": "mdi:camera",
                "entity_category": "diagnostic",
                "device": dev,
            }
        ),
        retain=True,
    )
    mqtt_client.publish(f"{DEVICE_ID}/camera_count", len(cams), retain=True)

    for cam in cams:
        mqtt_client.publish(
            f"homeassistant/binary_sensor/show-{cam.ha_name}/config",
            json.dumps(
                {
                    "name": f"Show {cam.name}".title(),
                    "state_topic": f"{cam.ha_name}/show",
                    "device_class": "occupancy",
                    "uniq_id": f"show-{cam.ha_name}",
                    "availability_topic": lwt,
                    "native_value": "boolean",
                    "payload_off": False,
                    "payload_on": True,
                    "device": dev,
                }
            ),
            retain=True,
        )
        mqtt_client.publish(f"{cam.ha_name}/show", False, retain=True)
        for item in cam.mqtt:
            mqtt_client.publish(
                f"homeassistant/sensor/{cam.ha_name}-{item}/config",
                json.dumps(
                    {
                        "name": f"{cam.name} {item} Count".title(),
                        "state_topic": f"{cam.ha_name}/{item}/count",
                        "state_class": "measurement",
                        "uniq_id": f"{cam.ha_name}-{item}",
                        "availability_topic": lwt,
                        "icon": mqtt_icons.get(item, f"mdi:{item}"),
                        "native_value": "int",
                        "device": dev,
                    }
                ),
                retain=True,
            )
            mqtt_client.publish(f"{cam.ha_name}/{item}/count", 0, retain=True)

    sd.notify("READY=1")
    sd.notify("STATUS=Running")
    cleanup_time = datetime(1970, 1, 1, 0, 0, 0)
    GracefulKiller()
    global kill_now
    while not kill_now:
        sd.notify("WATCHDOG=1")
        if mqtt_client._reconnect_deadline is not None:
            if time.monotonic() > mqtt_client._reconnect_deadline:
                log.error("MQTT reconnect failed after 5 minutes, shutting down")
                break
        start_time = timer()
        prediction_time = 0.0
        notify_time = 0.0
        capture_futures = []
        messages = []
        log_line = ""
        for cam in filter(lambda cam: cam.ftp_path, cams):
            try:
                capture_futures.append(async_pool.submit(cam.poll))
            except KeyboardInterrupt:
                return
            except requests.exceptions.ConnectionError:
                log.warning("cam:%s poll: %s", cam.name, sys.exc_info()[1])

        count = 0
        try:
            for f in concurrent.futures.as_completed(capture_futures, timeout=180):
                try:
                    cam = f.result()
                    if cam:
                        p, n, m = detect(
                            cam,
                            color_model,
                            grey_model,
                            vehicle_model,
                            config,
                            ha,
                        )
                        prediction_time += p
                        notify_time += n
                        messages.append(m)
                        count += 1
                except KeyboardInterrupt:
                    return
                except Exception:
                    log.exception("Error in detection pipeline")
        except concurrent.futures.TimeoutError:
            log.warning("Camera poll timed out after 180s, continuing")

        if count == 0:
            # scan each camera
            for cam in filter(
                lambda cam: (datetime.now() - cam.prior_time).total_seconds()
                > cam.interval,
                cams,
            ):
                try:
                    capture_futures.append(async_pool.submit(cam.capture))
                    count += 1
                except KeyboardInterrupt:
                    return
                except requests.exceptions.ConnectionError:
                    log.warning(
                        "cam:%s ConnectionError: %s", cam.name, sys.exc_info()[1]
                    )
            if count > 0:
                log_line = "Snapshotting "

            try:
                for f in concurrent.futures.as_completed(capture_futures, timeout=180):
                    try:
                        cam = f.result()
                        if cam:
                            p, n, m = detect(
                                cam,
                                color_model,
                                grey_model,
                                vehicle_model,
                                config,
                                ha,
                            )
                            prediction_time += p
                            notify_time += n
                            messages.append(m)
                    except KeyboardInterrupt:
                        return
                    except Exception:
                        log.exception("Error in detection pipeline")
            except concurrent.futures.TimeoutError:
                log.warning("Camera capture timed out after 180s, continuing")
        else:
            log_line = "Reading "

        end_time = timer()
        if count > 0:
            log_line += ",".join(sorted(messages))
            log_line += ".. completed in %.2fs, spent %.2fs predicting" % (
                (end_time - start_time),
                prediction_time,
            )
            if notify_time > 0:
                log_line += ", %.2fs notifying" % (notify_time)
        if len(log_line) > 0:
            log.info(log_line)
        if prediction_time < 0.1:
            if datetime.now() - cleanup_time > timedelta(minutes=15):
                log.debug("Cleaning up")
                for cam in filter(lambda cam: cam.ftp_path, cams):
                    cleanup(cam.ftp_path)
                    cam.globber = None
                cleanup_time = datetime.now()
            else:
                time.sleep(1.0)
        if "once" in detector_config:
            break

    # set item counts to unavailable
    for cam in cams:
        for item in cam.mqtt:
            mqtt_client.publish(f"{cam.name}/{item}/count", None, retain=False)
            mqtt_client.publish(f"{cam.ha_name}/{item}/count", None, retain=False)
        del cam
    # graceful shutdown
    log.info("Graceful shutdown initiated")
    mqtt_client.disconnect()  # disconnect gracefully
    mqtt_client.loop_stop()  # stops network loop
    # Models are cleaned up automatically at exit via atexit handler


if __name__ == "__main__":
    faulthandler.register(signal.SIGUSR1)
    # python 3.7 is asyncio.run()
    parser = argparse.ArgumentParser(description="Process cameras")
    parser.add_argument("--trt", action="store_true")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--config_file", nargs="?", default="config.txt")

    handlers = [
        logging.StreamHandler(),
        # logging.FileHandler("/var/log/aicam.log"),
    ]
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(name)-5s %(levelname)-4s %(message)s",
        datefmt="%b-%d %H:%M",
        handlers=handlers,
    )
    logging.getLogger("urllib3.connectionpool").setLevel(logging.INFO)
    logging.getLogger("detect").setLevel(logging.INFO)
    mlog.setLevel(logging.INFO)
    log.info("Starting")
    args = parser.parse_args()
    asyncio.get_event_loop().run_until_complete(main(args))
    log.info("Graceful shutdown complete")
