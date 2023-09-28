"""Useful functions."""
import os as _os
import importlib as _importlib
from collections import namedtuple as _namedtuple
from functools import partial as _partial
import pickle as _pickle
import subprocess as _subprocess
import pkg_resources as _pkg_resources
from types import ModuleType as _ModuleType

import h5py as _h5py
import gzip as _gzip
import numpy as _np

_HASSIRIUSPY = False
if _importlib.util.find_spec('siriuspy'):
    from siriuspy.namesys import SiriusPVName as _SiriusPVName
    _HASSIRIUSPY = True


def generate_random_numbers(n_part, dist_type='exp', cutoff=3):
    """Generate random numbers with a cutted off dist_type distribution.

    Inputs:
        n_part = size of the array with random numbers
        dist_type = assume values 'exponential', 'normal' or 'uniform'.
        cutoff = where to cut the distribution tail.
    """
    dist_type = dist_type.lower()
    if dist_type in 'exponential':
        func = _partial(_np.random.exponential, 1)
    elif dist_type in 'normal':
        func = _np.random.randn
    elif dist_type in 'uniform':
        func = _np.random.rand
    else:
        raise NotImplementedError('Distribution type not implemented yet.')

    numbers = func(n_part)
    above, *_ = _np.asarray(_np.abs(numbers) > cutoff).nonzero()
    while above.size:
        parts = func(above.size)
        indcs = _np.abs(parts) > cutoff
        numbers[above[~indcs]] = parts[~indcs]
        above = above[indcs]

    if dist_type in 'uniform':
        numbers -= 1/2
        numbers *= 2
    return numbers


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
    field_names = [f.replace(' ', '_') for f in field_names]
    return _namedtuple(name, field_names)(*values)


def is_gzip_file(fname):
    """Check if file is compressed with gzip.

    Args:
        fname (str): filename.

    Returns:
        bool: whether file is compressed with gzip.
    """
    # thanks to https://stackoverflow.com/questions/3703276/how-to-tell-if-a-file-is-gzip-compressed
    with open(fname, 'rb') as fil:
        return fil.read(2) == b'\x1f\x8b'


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
    if not fname.endswith('.pickle'):
        fname += '.pickle'
    if not overwrite and _os.path.isfile(fname):
        raise FileExistsError(f'file {fname} already exists.')
    if makedirs:
        dirname = _os.path.dirname(fname)
        if not _os.path.exists(dirname):
            _os.makedirs(dirname)
    with open(fname, 'wb') as fil:
        _pickle.dump(data, fil)


def load_pickle(fname):
    """Load ".pickle" file.

    Args:
        fname (str): Name of the file to load. May or may not contain the
            ".pickle" extension.

    Returns:
        data (any builtin type): content of file as a python object.

    """
    if not fname.endswith('.pickle'):
        fname += '.pickle'

    func = _gzip.open if is_gzip_file(fname) else open

    with func(fname, 'rb') as fil:
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
            saved in compressed format, using gzip library. Defaults to False.

    Raises:
        FileExistsError: in case `overwrite` is `False` and file exists.

    """
    comp_ = 'gzip' if compress else None

    if not fname.endswith(('.h5', '.hd5', '.hdf5', '.hdf')):
        fname += '.h5'

    if not overwrite and _os.path.isfile(fname):
        raise FileExistsError(f'file {fname} already exists.')
    elif _os.path.isfile(fname):
        _os.remove(fname)

    if makedirs:
        dirname = _os.path.dirname(fname)
        if not _os.path.exists(dirname):
            _os.makedirs(dirname)

    with _h5py.File(fname, 'w') as fil:
        _save_recursive_hdf5(fil, '/', data, comp_)


def load_hdf5(fname):
    """Load HDF5 file.

    Args:
        fname (str): Name of the file to load. If the extension of the file is
            ".h5" fname may not contain the extension. In other cases, where
            the extension may be ".hdf5", ".hdf", ".hd5", etc., it is
            mandatory that fname has the extension of the file.

    Returns:
        data (any builtin type): content of file as a python object.

    """
    if not fname.endswith(('.h5', '.hd5', '.hdf5', '.hdf')):
        fname += '.h5'

    with _h5py.File(fname, 'r') as fil:
        return _load_recursive_hdf5(fil)


def repo_info(repo_path):
    """Get repository information.

    Args:
        repo_path (str): Repository path.

    Returns:
        repo_info (dict): Repository info.
            path, active_branch, last_tag, last_commit, is_dirty.

    """
    info = dict()
    err = _subprocess.CalledProcessError
    out = _subprocess.check_output
    # save initial directory and go to repository directory
    init_dir = _os.getcwd()
    _os.chdir(repo_path)
    try:
        cmd = ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
        active_branch = out(cmd, universal_newlines=True).strip()
    except err:
        print(f'Repository path not found: {repo_path}.')
        return info
    try:
        cmd = ['git', 'describe', '--tags', '--abbrev=0']
        last_tag = out(cmd, universal_newlines=True).strip()
        cmd = ['git', 'rev-list', '-n', '1', last_tag]
        last_tag_commit = out(cmd, universal_newlines=True).strip()[:7]
    except err:
        last_tag = ''
        last_tag_commit = ''
    cmd = ['git', 'rev-parse', 'HEAD']
    last_commit = out(cmd, universal_newlines=True).strip()[:7]
    cmd = ['git', 'status', '--short']
    is_dirty = out(cmd, universal_newlines=True).strip() != ''
    # return to initial directory
    _os.chdir(init_dir)

    info['path'] = repo_path
    info['active_branch'] = active_branch
    info['last_tag'] = last_tag
    info['last_tag_commit'] = last_tag_commit
    info['last_commit'] = last_commit
    info['is_dirty'] = is_dirty
    return info


def get_path_from_package(package):
    """Return the directory where package is installed.

    Args:
        package (str or module): Package name or module

    Raises:
        ValueError: If package argument type is different from str or module

    Returns:
        location (str): Package installation directory
        version (str) : Package installation version

    """
    if isinstance(package, str):
        pkg = package
    elif isinstance(package, _ModuleType):
        pkg = package.__package__
    else:
        raise ValueError('Invalid package type, must be str or module')
    dist = _pkg_resources.get_distribution(pkg)
    return dist.location, dist.version


def is_git_repo(path):
    """."""
    try:
        cmd = ['git', 'rev-parse', '--is-inside-work-tree']
        null = _subprocess.DEVNULL
        _subprocess.run(cmd, cwd=path, check=True, stdout=null, stderr=null)
        return True
    except _subprocess.CalledProcessError:
        return False


def get_package_string(package):
    """."""
    path, ver = get_path_from_package(package)
    repo_str = ''
    if is_git_repo(path):
        info = repo_info(path)
        if info['last_tag']:
            repo_str += f"{info['last_tag']:s}"
        if info['last_tag_commit'] != info['last_commit']:
            repo_str += '+' if info['last_tag'] else ''
            repo_str += f"{info['last_commit']:s}"
        if info['is_dirty']:
            repo_str += f"+dirty"
    else:
        repo_str += f'{ver:s}'
    return repo_str


# ------------------------- HELPER METHODS ------------------------------------
_TYPES2SAVE = (int, float, complex, str, bytes, bool)
_STRTYPES = {typ.__name__ for typ in _TYPES2SAVE}
_NPTYPES = (_np.int_, _np.float, _np.complex_, _np.bool_)


def _save_recursive_hdf5(fil, path, obj, compress):
    typ = type(obj)
    if isinstance(obj, _np.ndarray):
        fil.create_dataset(path, data=obj, compression=compress)
    elif isinstance(obj, str):  # Handle SiriusPVName
        fil[path] = str(obj)
    elif isinstance(obj, _TYPES2SAVE + _NPTYPES):
        fil[path] = obj
    elif obj is None:
        fil[path] = 'None'
    elif typ in {list, tuple, set}:
        path = path.strip('/') + '/'
        for i, el in enumerate(obj):
            _save_recursive_hdf5(fil, path + f'{i:d}', el, compress)
    elif typ == dict:
        path = path.strip('/') + '/'
        for key, item in obj.items():
            _save_recursive_hdf5(fil, path + key, item, compress)
    else:
        raise TypeError('Data type not valid: '+typ.__name__+'.')
    fil[path].attrs['type'] = typ.__name__


def _load_recursive_hdf5(fil):
    """."""
    typ = fil.attrs['type']
    if typ == 'list':
        return [_load_recursive_hdf5(fil[str(i)]) for i in range(len(fil))]
    elif typ == 'tuple':
        return tuple([
            _load_recursive_hdf5(fil[str(i)]) for i in range(len(fil))])
    elif typ == 'set':
        return {_load_recursive_hdf5(fil[str(i)]) for i in range(len(fil))}
    elif typ == 'dict':
        return {k: _load_recursive_hdf5(obj) for k, obj in fil.items()}
    elif typ == 'NoneType':
        return None
    elif typ == 'ndarray':
        return fil[()]
    elif typ == 'str':
        return fil[()].decode()
    elif _HASSIRIUSPY and typ == 'SiriusPVName':
        return _SiriusPVName(fil[()].decode())
    elif typ in _STRTYPES:
        exec('type_ = '+typ)
        return locals()['type_'](fil[()])
    else:
        exec('type_ = _np.'+typ)
        return locals()['type_'](fil[()])
