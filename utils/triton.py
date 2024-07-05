# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""Utils to interact with the Triton Inference Server."""

import typing
from urllib.parse import urlparse

import torch


class TritonRemoteModel:
    """
    A wrapper over a model served by the Triton Inference Server.

    It can be configured to communicate over GRPC or HTTP. It accepts Torch Tensors as input and returns them as
    outputs.
    """

    def __init__(self, url: str):
        """
        Initializes the `TritonRemoteModel` object that wraps a model served by the Triton Inference Server.

        Args:
            url (str): Fully qualified address of the Triton server,
                e.g., "grpc://localhost:8000" or "http://localhost:8000".

        Returns:
            None

        Raises:
            ValueError: If the provided URL scheme is neither GRPC nor HTTP.

        Notes:
            - The class communicates with the Triton server over either GRPC or HTTP based on the provided URL scheme.
            - Model metadata, including names and input shapes, are parsed and stored for future inference operations.

        Example:
            ```python
            triton_model = TritonRemoteModel("grpc://localhost:8000")
            ```
        """

        parsed_url = urlparse(url)
        if parsed_url.scheme == "grpc":
            from tritonclient.grpc import InferenceServerClient, InferInput

            self.client = InferenceServerClient(parsed_url.netloc)  # Triton GRPC client
            model_repository = self.client.get_model_repository_index()
            self.model_name = model_repository.models[0].name
            self.metadata = self.client.get_model_metadata(self.model_name, as_json=True)

            def create_input_placeholders() -> typing.List[InferInput]:
                return [
                    InferInput(i["name"], [int(s) for s in i["shape"]], i["datatype"]) for i in self.metadata["inputs"]
                ]

        else:
            from tritonclient.http import InferenceServerClient, InferInput

            self.client = InferenceServerClient(parsed_url.netloc)  # Triton HTTP client
            model_repository = self.client.get_model_repository_index()
            self.model_name = model_repository[0]["name"]
            self.metadata = self.client.get_model_metadata(self.model_name)

            def create_input_placeholders() -> typing.List[InferInput]:
                return [
                    InferInput(i["name"], [int(s) for s in i["shape"]], i["datatype"]) for i in self.metadata["inputs"]
                ]

        self._create_input_placeholders_fn = create_input_placeholders

    @property
    def runtime(self):
        """
        Returns the model runtime.

        Args:
            None

        Returns:
            str: The runtime environment of the model, such as 'GRPC' or 'HTTP', depending on the communication protocol
            configured during initialization.

        Notes:
            - This property fetches and returns the runtime protocol used by the Triton Inference Server, which is set
              based on the URL scheme provided during initialization.
            - The function extracts the communication protocol from the URL provided and returns it as a string.

        Example:
            ```python
            model = TritonRemoteModel('grpc://localhost:8000')
            print(model.runtime)  # Output: 'GRPC'
            ```
        """
        return self.metadata.get("backend", self.metadata.get("platform"))

    def __call__(self, *args, **kwargs) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]]:
        """
        Invokes the model hosted on Triton Inference Server and processes the input and output tensors.

        Args:
            *args: Positional arguments that match the order of model inputs.
            **kwargs: Keyword arguments that match the model input names.

        Returns:
            torch.Tensor | tuple[torch.Tensor, ...]: The output tensors from the model.

        Example:
            ```python
            model = TritonRemoteModel('grpc://localhost:8000')
            output = model(torch.tensor([1.0, 2.0, 3.0]))
            ```
        """
        inputs = self._create_inputs(*args, **kwargs)
        response = self.client.infer(model_name=self.model_name, inputs=inputs)
        result = []
        for output in self.metadata["outputs"]:
            tensor = torch.as_tensor(response.as_numpy(output["name"]))
            result.append(tensor)
        return result[0] if len(result) == 1 else result

    def _create_inputs(self, *args, **kwargs):
        """
        Generates model inputs from args or kwargs by creating the necessary placeholders and populating them with data.

        Args:
            *args: Variable length argument list representing the ordered inputs to the model. Each entry is a `torch.Tensor`.
            **kwargs: Keyword arguments representing the named inputs to the model. The keys should match the model's input
                names and the values should be `torch.Tensor`.

        Returns:
            typing.List[InferInput]: List of InferInput objects populated with the data from the provided tensors.

        Raises:
            RuntimeError: If neither args nor kwargs are provided, or if both are provided.
            RuntimeError: If the number of args does not match the expected number of model inputs.
        """
        args_len, kwargs_len = len(args), len(kwargs)
        if not args_len and not kwargs_len:
            raise RuntimeError("No inputs provided.")
        if args_len and kwargs_len:
            raise RuntimeError("Cannot specify args and kwargs at the same time")

        placeholders = self._create_input_placeholders_fn()
        if args_len:
            if args_len != len(placeholders):
                raise RuntimeError(f"Expected {len(placeholders)} inputs, got {args_len}.")
            for input, value in zip(placeholders, args):
                input.set_data_from_numpy(value.cpu().numpy())
        else:
            for input in placeholders:
                value = kwargs[input.name]
                input.set_data_from_numpy(value.cpu().numpy())
        return placeholders
