from functools import wraps
from threading import Lock


def non_blocking_lock(fn):
    """Decorator. Prevents the function from being called multiple times simultaneously.

    If thread A is executing the function and thread B attempts to call the
    function, thread B will immediately receive a return value of None instead.
    """

    lock = Lock()

    @wraps(fn)
    def locker(*args, **kwargs):
        if lock.acquire(False):
            try:
                return fn(*args, **kwargs)
            finally:
                lock.release()

    return locker
