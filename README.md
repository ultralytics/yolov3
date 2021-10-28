How to run the yolv3 with trained weights using docker.

1. Add video to Assignment3/Data directory.
2. Add frames to infer json to Assignment3/JSON directory.
3. Add trained weights to Assignment3/Weights directory.

Install docker to run the yolov3 in a docker container.

I have written a bash script to run the docker container.

In the root folder, run "bash run.sh" in the terminal to run yolov3 in the container.

Once the yolov3 model has done its inference, it will copy the results to Assignment/Results directory.
