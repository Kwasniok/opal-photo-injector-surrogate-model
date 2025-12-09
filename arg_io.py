"""
Module for conversion of command-line arguments between different representations:
- list of strings (argv)
- dictionary (argd)
- file storage"""

from typing import Sequence
import json
import ast

ArgType = str | int | float
ArgDict = dict[str, ArgType | Sequence[ArgType]]


def argd_to_argv(argd: ArgDict) -> list[str]:
    """
    Converts a dictionary of command-line arguments to a list of strings.

    DICT FORMAT:
    {
        "key1": value1,
        "key2": [value2a, value2b, ...],
        ...
    }
    where values are of the type str, int or float.

    LIST FORMAT:
    list of:
    --key value1 value2 ... valueN
    where values are of the type str, int or float.
    """
    argv: list[str] = []
    for k, v in argd.items():
        argv.append(f"--{k}")
        if isinstance(v, list | tuple):
            argv.extend(map(str, v))
        else:
            argv.append(str(v))
    return argv


def argv_to_file(filename: str, argv: list[str]) -> None:
    """
    Saves the command-line arguments represented by the list `argv` to a file.

    Simple string representation of list.

    LIST FORMAT:
    list of:
    --key value1 value2 ... valueN
    where values are of the type str, int or float.
    """
    with open(filename, "w") as f:
        f.write(str(argv))


def argd_to_file(filename: str, argd: ArgDict) -> None:
    """
    Saves the command-line arguments represented by the dictionary `d` to a file.

    Simple string representation of list.

    DICT FORMAT:
    {
        "key1": value1,
        "key2": [value2a, value2b, ...],
        ...
    }
    where values are of the type str, int or float.
    """
    argv = argd_to_argv(argd)
    argv_to_file(filename, argv)


def argv_to_argd(argv: list[str]) -> ArgDict:
    """
    Returns a dictionary representing the command-line arguments in `argv`.

    LIST FORMAT:
    list of:
    --key value1 value2 ... valueN
    where values are of the type str, int or float

    DICT FORMAT:
    {
        "key1": value1,
        "key2": [value2a, value2b, ...],
        ...
    }
    where values are of the type str, int or float.
    """
    result: dict[str, list[str]] = {}
    key: str | None = None
    for arg in argv:
        if arg.startswith("--"):
            key = arg.lstrip("-")
            result[key] = []
        else:
            if key is None:
                raise ValueError(f"Argument {arg} has no preceding key!")
            result[key].append(arg)

    def parse(value: str) -> ArgType:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    parsed_result: dict[str, list[ArgType]] = {
        k: list(map(parse, vs)) for k, vs in result.items()
    }

    # simplify singletons to values
    def simplify(vs: list[ArgType]) -> ArgType | list[ArgType]:
        if len(vs) == 1:
            return vs[0]
        return vs

    return {k: simplify(vs) for k, vs in parsed_result.items()}


def argv_from_file(filename: str) -> list[str]:
    """
    Loads command-line arguments from a file.

    Simple string representation of list.

    LIST FORMAT:
    list of:
    --key value1 value2 ... valueN
    where values are of the type str, int or float

    """
    with open(filename, "r") as f:
        content = f.read()
    return ast.literal_eval(content)


def argd_from_file(filename: str) -> ArgDict:
    """
    Loads command-line arguments from a file and returns them as a dictionary.

    DICT FORMAT:
    {
        "key1": value1,
        "key2": [value2a, value2b, ...],
        ...
    }
    where values are of the type str, int or float.
    """
    argv = argv_from_file(filename)
    return argv_to_argd(argv)
