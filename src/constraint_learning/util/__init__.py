from . import sampling  # noqa: F401


def to_dict(input):
    """To get around not being able to have empty dicts as default values.

    Args:
        input (None or dict): The value to convert to a dict

    Returns:
        dict: The converted dict. An input of None will be returned with an
        empty dict.
    """
    if input is None:
        return {}
    else:
        return dict(input)
