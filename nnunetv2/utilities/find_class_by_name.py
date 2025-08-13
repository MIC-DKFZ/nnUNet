import importlib
import pkgutil
import sys
from contextlib import contextmanager
from os.path import abspath, join


from batchgenerators.utilities.file_and_folder_operations import *


@contextmanager
def temporarily_extend_syspath(path: str):
    """
    Context manager to temporarily add a directory to sys.path.
    If the path is not already in sys.path, it gets added and then removed on exit.
    """
    path = abspath(path)
    already_present = path in sys.path
    if not already_present:
        sys.path.insert(0, path)
    try:
        yield
    finally:
        if not already_present and path in sys.path:
            sys.path.remove(path)


def recursive_find_python_class(
    folder: str,
    class_name: str,
    current_module: str | None,
    base_folder: str | None = None,
    verbose: bool = False,
):
    """
    Recursively searches for a class with the given name in a Python package directory tree.
    Parameters
    ----------
    folder : str
        The directory path to start the search in.
    class_name : str
        The name of the class to search for.
    current_module : str or None
        The dotted Python module path corresponding to `folder`.
        E.g., "my_package.subpackage". Can be None if starting from a flat folder.
    base_folder : str or None, optional
        The root directory that should be temporarily added to sys.path to allow imports.
        If None, `folder` is used.
    verbose : bool, optional
        If True, print progress during the search.
    Returns
    -------
    type or None
        The found class object, or None if not found.
    """
    if base_folder is None:
        base_folder = folder

    with temporarily_extend_syspath(base_folder):
        if verbose:
            print(
                f"Searching for class {class_name} in folder {folder} with current module {current_module}"
            )

        # Search modules (non-packages) in the folder
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            if not ispkg:
                search_module = (
                    modname if current_module is None else f"{current_module}.{modname}"
                )
                if verbose:
                    print(f"  Inspecting module: {search_module}")
                try:
                    m = importlib.import_module(search_module)
                    if hasattr(m, class_name):
                        if verbose:
                            print(f"Found class {class_name} in {search_module}")
                        return getattr(m, class_name)
                except Exception as e:
                    print(f"Warning: Could not import module {search_module}: {e}")

        # Recurse into subpackages
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            if ispkg:
                next_folder = join(folder, modname)
                next_module = (
                    modname if current_module is None else f"{current_module}.{modname}"
                )
                result = recursive_find_python_class(
                    next_folder,
                    class_name,
                    current_module=next_module,
                    base_folder=base_folder,
                    verbose=verbose,
                )
                if result is not None:
                    return result

    return None
