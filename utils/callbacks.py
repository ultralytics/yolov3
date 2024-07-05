# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""Callback utils."""

import threading


class Callbacks:
    """" Handles all registered callbacks for YOLOv3 Hooks."""

    def __init__(self):
        """
        Initializes a Callbacks object to manage YOLOv3 training event hooks, facilitating the registration and
        execution of callback functions for various training stages.

        Args:
            None

        Returns:
            None

        Notes:
            The `Callbacks` object contains a dictionary, `_callbacks`, where each key corresponds to a specific training event, and the values are lists that store callback functions to be triggered at respective events. This design enables modular and flexible execution of additional functionality during the training process.
        """
        self._callbacks = {
            "on_pretrain_routine_start": [],
            "on_pretrain_routine_end": [],
            "on_train_start": [],
            "on_train_epoch_start": [],
            "on_train_batch_start": [],
            "optimizer_step": [],
            "on_before_zero_grad": [],
            "on_train_batch_end": [],
            "on_train_epoch_end": [],
            "on_val_start": [],
            "on_val_batch_start": [],
            "on_val_image_end": [],
            "on_val_batch_end": [],
            "on_val_end": [],
            "on_fit_epoch_end": [],  # fit = train + val
            "on_model_save": [],
            "on_train_end": [],
            "on_params_update": [],
            "teardown": [],
        }
        self.stop_training = False  # set True to interrupt training

    def register_action(self, hook, name="", callback=None):
        """
        Register a new action to a specified callback hook.

        Args:
            hook (str): The name of the callback hook to which the action will be registered. Must be one of
                the predefined hooks within the `self._callbacks` dictionary.
            name (str, optional): A human-readable name for the action. This is useful for identifying the
                action during debugging or logging.
            callback (Callable): A callable function or method that will be executed when the specified
                hook is triggered.

        Returns:
            None: This function does not return any value.

        Raises:
            AssertionError: If the specified `hook` is not found within the `self._callbacks` dictionary.
            AssertionError: If the `callback` is not a callable object.

        Examples:
            ```python
            def my_callback_function():
                print("Callback executed!")

            callbacks = Callbacks()
            callbacks.register_action(hook='on_train_start', name='print_callback', callback=my_callback_function)
            ```

        - Note: The `hook` parameter should match one of the keys in the `self._callbacks` dictionary, and
          the `callback` must be a callable function or method.
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        assert callable(callback), f"callback '{callback}' is not callable"
        self._callbacks[hook].append({"name": name, "callback": callback})

    def get_registered_actions(self, hook=None):
        """
        Returns all the registered actions for a specified callback hook or for all hooks if no hook is specified.

        Args:
            hook (str | None): The name of the callback hook to retrieve actions for. If None, retrieves actions
                for all hooks.

        Returns:
            dict (str, list of dict): A dictionary mapping each callback hook name to a list of registered actions.
                Each action is represented as a dictionary containing 'name' and 'callback'.

        Examples:
            To retrieve actions for a specific hook:
            ```python
            callbacks = Callbacks()
            actions = callbacks.get_registered_actions(hook="on_train_start")
            ```

            To retrieve actions for all hooks:
            ```python
            callbacks = Callbacks()
            actions = callbacks.get_registered_actions()
            ```
        """
        return self._callbacks[hook] if hook else self._callbacks

    def run(self, hook, *args, thread=False, **kwargs):
        """
        Runs all registered callbacks for a specified hook, optionally in a separate thread.

        Args:
            hook (str): The name of the hook to trigger callbacks for.
            args (tuple): Positional arguments to pass to the callbacks.
            thread (bool): If True, runs callbacks in a daemon thread. Defaults to False.
            kwargs (dict): Keyword arguments to pass to the callbacks.

        Returns:
            None

        Raises:
            AssertionError: If the specified hook is not found in the registered callbacks.

        Example:
            ```python
            callbacks = Callbacks()
            callbacks.register_action('on_train_start', device_setup)
            callbacks.run('on_train_start', thread=True)
            ```

        Note:
            This function ensures that all callbacks registered under the specified hook are executed. If the `thread` flag is set to
            True, each callback is run in a separate daemon thread, which allows for non-blocking execution and concurrent operations.
        """

        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        for logger in self._callbacks[hook]:
            if thread:
                threading.Thread(target=logger["callback"], args=args, kwargs=kwargs, daemon=True).start()
            else:
                logger["callback"](*args, **kwargs)
