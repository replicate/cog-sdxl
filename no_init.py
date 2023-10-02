import contextlib
import contextvars
import threading
from typing import (
    Callable,
    ContextManager,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
)

import torch

__all__ = ["no_init_or_tensor"]



Model = TypeVar("Model")


def no_init_or_tensor(
    loading_code: Optional[Callable[..., Model]] = None
) -> Union[Model, ContextManager]:
    """
    Suppress the initialization of weights while loading a model.

    Can either directly be passed a callable containing model-loading code,
    which will be evaluated with weight initialization suppressed,
    or used as a context manager around arbitrary model-loading code.

    Args:
        loading_code: Either a callable to evaluate
            with model weight initialization suppressed,
            or None (the default) to use as a context manager.

    Returns:
        The return value of `loading_code`, if `loading_code` is callable.

        Otherwise, if `loading_code` is None, returns a context manager
        to be used in a `with`-statement.

    Examples:
        As a context manager::

            from transformers import AutoConfig, AutoModelForCausalLM
            config = AutoConfig("EleutherAI/gpt-j-6B")
            with no_init_or_tensor():
                model = AutoModelForCausalLM.from_config(config)

        Or, directly passing a callable::

            from transformers import AutoConfig, AutoModelForCausalLM
            config = AutoConfig("EleutherAI/gpt-j-6B")
            model = no_init_or_tensor(lambda: AutoModelForCausalLM.from_config(config))
    """
    if loading_code is None:
        return _NoInitOrTensorImpl.context_manager()
    elif callable(loading_code):
        with _NoInitOrTensorImpl.context_manager():
            return loading_code()
    else:
        raise TypeError(
            "no_init_or_tensor() expected a callable to evaluate,"
            " or None if being used as a context manager;"
            f' got an object of type "{type(loading_code).__name__}" instead.'
        )


class _NoInitOrTensorImpl:
    # Implementation of the thread-safe, async-safe, re-entrant context manager
    # version of no_init_or_tensor().
    # This class essentially acts as a namespace.
    # It is not instantiable, because modifications to torch functions
    # inherently affect the global scope, and thus there is no worthwhile data
    # to store in the class instance scope.
    _MODULES = (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)
    _MODULE_ORIGINALS = tuple((m, m.reset_parameters) for m in _MODULES)
    _ORIGINAL_EMPTY = torch.empty

    is_active = contextvars.ContextVar("_NoInitOrTensorImpl.is_active", default=False)
    _count_active: int = 0
    _count_active_lock = threading.Lock()

    @classmethod
    @contextlib.contextmanager
    def context_manager(cls):
        if cls.is_active.get():
            yield
            return

        with cls._count_active_lock:
            cls._count_active += 1
            if cls._count_active == 1:
                for mod in cls._MODULES:
                    mod.reset_parameters = cls._disable(mod.reset_parameters)
                # When torch.empty is called, make it map to meta device by replacing
                # the device in kwargs.
                torch.empty = cls._ORIGINAL_EMPTY
        reset_token = cls.is_active.set(True)

        try:
            yield
        finally:
            cls.is_active.reset(reset_token)
            with cls._count_active_lock:
                cls._count_active -= 1
                if cls._count_active == 0:
                    torch.empty = cls._ORIGINAL_EMPTY
                    for mod, original in cls._MODULE_ORIGINALS:
                        mod.reset_parameters = original

    @staticmethod
    def _disable(func):
        def wrapper(*args, **kwargs):
            # Behaves as normal except in an active context
            if not _NoInitOrTensorImpl.is_active.get():
                return func(*args, **kwargs)

        return wrapper

    __init__ = None
