import importlib
import os
import pkgutil
import sys
from contextlib import contextmanager, ExitStack
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


def _module_originates_from_path(module, path: str) -> bool:
    path = abspath(path)

    module_file = getattr(module, "__file__", None)
    if module_file is not None:
        try:
            return os.path.commonpath((path, abspath(module_file))) == path
        except ValueError:
            return False

    module_path = getattr(module, "__path__", None)
    if module_path is not None:
        for candidate in module_path:
            try:
                if os.path.commonpath((path, abspath(candidate))) == path:
                    return True
            except ValueError:
                continue

    return False


@contextmanager
def temporarily_cleanup_imports_from_path(path: str):
    """
    Removes modules imported from `path` when leaving the context.

    This prevents one external trainer directory from poisoning later lookups through
    reused entries in sys.modules.
    """
    path = abspath(path)
    previously_loaded = {
        name
        for name, module in sys.modules.items()
        if module is not None and _module_originates_from_path(module, path)
    }
    try:
        yield
    finally:
        for name, module in list(sys.modules.items()):
            if (
                name not in previously_loaded
                and module is not None
                and _module_originates_from_path(module, path)
            ):
                sys.modules.pop(name, None)


def _recursive_find_python_class(folder: str, class_name: str, current_module: str | None, verbose: bool = False):
    # Search modules (non-packages) in the folder
    for importer, modname, ispkg in pkgutil.iter_modules([folder]):
        if not ispkg:
            search_module = (
                modname if current_module is None else f"{current_module}.{modname}"
            )
            if verbose:
                print(f"  Inspecting module: {search_module}")
            m = importlib.import_module(search_module)
            if hasattr(m, class_name):
                if verbose:
                    print(f"Found class {class_name} in {search_module}")
                return getattr(m, class_name)

    # Recurse into subpackages
    for importer, modname, ispkg in pkgutil.iter_modules([folder]):
        if ispkg:
            next_folder = join(folder, modname)
            next_module = (
                modname if current_module is None else f"{current_module}.{modname}"
            )
            result = _recursive_find_python_class(
                next_folder,
                class_name,
                current_module=next_module,
                verbose=verbose,
            )
            if result is not None:
                return result

    return None


def recursive_find_python_class(
    folder: str,
    class_name: str,
    current_module: str | None,
    base_folder: str | None = None,
    verbose: bool = False,
    cleanup_imports_from_base_folder: bool = False,
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

    with ExitStack() as stack:
        stack.enter_context(temporarily_extend_syspath(base_folder))
        if cleanup_imports_from_base_folder:
            stack.enter_context(temporarily_cleanup_imports_from_path(base_folder))

        if verbose:
            print(
                f"Searching for class {class_name} in folder {folder} with current module {current_module}"
            )

        return _recursive_find_python_class(
            folder,
            class_name,
            current_module=current_module,
            verbose=verbose,
        )
