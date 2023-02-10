import asyncio
from functools import partial


async def wrapBlocking(func, *args, **kwargs):
    loop = asyncio.get_event_loop()

    wrapped = partial(func, *args, **kwargs)

    return await loop.run_in_executor(None, wrapped)


def add_doc(value):
    """Wrap method to programatically add docstring."""

    def _doc(func):
        func.__doc__ = value.__doc__
        return func

    return _doc
