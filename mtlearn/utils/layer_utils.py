import sys
import inspect
from typing import Callable

def has_arg(function: Callable,
            argument: str
            ) -> bool:
    """
    Checks whether callable accepts given keyword argument

    Parameters
    ----------
    function: callable
      Function which which arguments we are checking

    argument: str
      Name of the argument

    Returns
    -------
    : bool
      True if user provided argument is one of the keyword arguments
      of the given function

    References
    ----------
    https://github.com/keras-team/keras/pull/7035/
    """
    if sys.version_info < (3, 3):
        arg_spec = inspect.getfullargspec(function)
        return argument in arg_spec.args
    else:
        signature = inspect.signature(function)
        return signature.parameters.get(argument) is not None
