"""Useful functions."""

import builtins as _builtins
import gzip as _gzip
import os as _os
import pickle as _pickle
from collections import namedtuple as _namedtuple

import h5py as _h5py
import numpy as _np


def get_namedtuple(name, field_names, values=None):
    """Return an instance of a namedtuple Class.

    Inputs:
        - name:  Defines the name of the Class (str).
        - field_names:  Defines the field names of the Class (iterable).
        - values (optional): Defines field values . If not given, the value of
            each field will be its index in 'field_names' (iterable).

    Raises ValueError if at least one of the field names are invalid.
    Raises TypeError when len(values) != len(field_names)
    """
    if values is None:
        values = range(len(field_names))
    field_names = [f.replace(" ", "_") for f in field_names]
    return _namedtuple(name, field_names)(*values)


def is_gzip_file(fname):
    """Check if file is compressed with gzip.

    Args:
        fname (str): filename.

    Returns:
        bool: whether file is compressed with gzip.
    """
    # thanks to https://stackoverflow.com/questions/3703276/how-to-tell-if-a-file-is-gzip-compressed
    with open(fname, "rb") as fil:
        return fil.read(2) == b"\x1f\x8b"


def save(data, fname, overwrite=False, makedirs=False, compress=False):
    """Save data to pickle or HDF5 file.

    Args:
        data (any builtin type): python object to be saved.
        fname (str): name of the file. If extension is not provided,
            '.pickle' will be added and a pickle file will be assumed.
            If provided, must be {'.pickle', .pkl} for pickle files or
            {'.h5', '.hdf5', '.hdf', '.hd5'} for HDF5 files.
        overwrite (bool, optional): Whether to overwrite existing file.
            Defaults to False.
        makedirs (bool, optional): create dir, if it does not exist.
            Defaults to False.
        compress (bool, optional): If True, the file will be saved in
            compressed format, using gzip library. Defaults to False.

    Raises:
        FileExistsError: in case `overwrite` is `False` and file exists.

    """
    _save = save_pickle
    if fname.endswith((".h5", ".hd5", ".hdf5", ".hdf")):
        _save = save_hdf5
    _save(data, fname, overwrite, makedirs, compress)


def load(fname):
    """Load and return data from pickle or HDF5 file.

    Args:
        fname (str): name of the file. If extension is not provided,
            '.pickle' will be added and a pickle file will be assumed.
            If provided, must be {'.pickle', '.pkl'} for pickle files or
            {'.h5', '.hdf5', '.hdf', '.hd5'} for HDF5 files.

    Returns:
         data (any builtin type): content of file as a python object.

    """
    _load = load_pickle
    if fname.endswith((".h5", ".hd5", ".hdf5", ".hdf")):
        _load = load_hdf5
    return _load(fname)


def save_pickle(data, fname, overwrite=False, makedirs=False, compress=False):
    """Save data to file in pickle format.

    Args:
        data (any builtin type): python object to be saved
        fname (str): name of the file to be saved. With or without ".pickle"."
        overwrite (bool, optional): whether to overwrite existing file.
            Defaults to False.
        makedirs (bool, optional): create dir, if it does not exist.
            Defaults to False.
        compress (bool, optional): If True, the file will be saved in
            compressed format, using gzip library. Defaults to False.

    Raises:
        FileExistsError: in case `overwrite` is `False` and file exists.

    """
    if not fname.endswith((".pickle", ".pkl")):
        fname += ".pickle"

    if not overwrite and _os.path.isfile(fname):
        raise FileExistsError(f"file {fname} already exists.")

    if makedirs:
        dirname = _os.path.dirname(fname)
        if not _os.path.exists(dirname):
            _os.makedirs(dirname)

    func = _gzip.open if compress else open
    with func(fname, "wb") as fil:
        _pickle.dump(data, fil)


def load_pickle(fname):
    """Load ".pickle" file.

    Args:
        fname (str): Name of the file to load. May or may not contain the
            ".pickle" extension.

    Returns:
        data (any builtin type): content of file as a python object.

    """
    if not fname.endswith((".pickle", ".pkl")):
        fname += ".pickle"

    func = _gzip.open if is_gzip_file(fname) else open

    with func(fname, "rb") as fil:
        data = _pickle.load(fil)
    return data


def save_hdf5(data, fname, overwrite=False, makedirs=False, compress=False):
    """Save data to HDF5 file.

    Args:
        data (any builtin type): python object to be saved
        fname (str): name of the file to be saved. With or without ".h5".
        overwrite (bool, optional): whether to overwrite existing file.
            Defaults to False.
        makedirs (bool, optional): create dir, if it does not exist.
            Defaults to False.
        compress (bool, optional): If True, the arrays in the file will be
            saved in compressed format, using hdf5 standard compression with
            the `gzip` filter. Defaults to False.

    Raises:
        FileExistsError: in case `overwrite` is `False` and file exists.

    """
    comp_ = "gzip" if compress else None

    if not fname.endswith((".h5", ".hd5", ".hdf5", ".hdf")):
        fname += ".h5"

    if not overwrite and _os.path.isfile(fname):
        raise FileExistsError(f"file {fname} already exists.")
    elif _os.path.isfile(fname):
        _os.remove(fname)

    if makedirs:
        dirname = _os.path.dirname(fname)
        if not _os.path.exists(dirname):
            _os.makedirs(dirname)

    with _h5py.File(fname, "w") as fil:
        _save_recursive_hdf5(fil, "/", data, comp_)


def load_hdf5(fname):
    """Load a HDF5 file as whole to the memory as a python object.

    Args:
        fname (str): Name of the file to load. If the extension of the file is
            ".h5" fname may not contain the extension. In other cases, where
            the extension may be ".hdf5", ".hdf", ".hd5", etc., it is
            mandatory that fname has the extension of the file.

    Returns:
        data (any builtin type): content of file as a python object.

    """
    if not fname.endswith((".h5", ".hd5", ".hdf5", ".hdf")):
        fname += ".h5"

    with _h5py.File(fname, "r") as fil:
        return _load_recursive_hdf5(fil)


# ------------------------- HELPER METHODS ------------------------------------
_BUILTINTYPES = (int, float, complex, str, bytes, bool)
_BUILTINNAMES = {typ.__name__ for typ in _BUILTINTYPES}
_NPTYPES = (_np.int_, _np.float_, _np.complex_, _np.bool_)


def _save_recursive_hdf5(fil, path, obj, compress):
    typ = type(obj)
    if isinstance(obj, _np.ndarray):
        fil.create_dataset(path, data=obj, compression=compress)
    elif isinstance(obj, str):  # Handle SiriusPVName
        fil[path] = str(obj)
    elif isinstance(obj, _BUILTINTYPES + _NPTYPES):
        fil[path] = obj
    elif obj is None:
        fil[path] = "None"
    elif typ in {list, tuple, set}:
        path = path.strip("/") + "/"
        for i, el in enumerate(obj):
            _save_recursive_hdf5(fil, path + f"{i:d}", el, compress)
    elif typ is dict:
        path = path.strip("/") + "/"
        for key, item in obj.items():
            _save_recursive_hdf5(fil, path + key, item, compress)
    else:
        raise TypeError("Data type not valid: " + typ.__name__ + ".")
    fil[path].attrs["type"] = typ.__name__


def _load_recursive_hdf5(fil):
    """."""
    typ = fil.attrs["type"]
    if typ == "list":
        return [_load_recursive_hdf5(fil[str(i)]) for i in range(len(fil))]
    elif typ == "tuple":
        return tuple(
            [_load_recursive_hdf5(fil[str(i)]) for i in range(len(fil))]
        )
    elif typ == "set":
        return {_load_recursive_hdf5(fil[str(i)]) for i in range(len(fil))}
    elif typ == "dict":
        return {k: _load_recursive_hdf5(obj) for k, obj in fil.items()}
    elif typ == "NoneType":
        return None
    elif typ == "ndarray":
        return fil[()]
    elif typ == "str":
        return fil[()].decode()
    elif typ == "SiriusPVName":
        return fil[()].decode()
    elif typ in _BUILTINNAMES:
        type_ = getattr(_builtins, typ)
        return type_(fil[()])
    else:
        # h5py automatically handles convertion of scalar numpy types
        return fil[()]
