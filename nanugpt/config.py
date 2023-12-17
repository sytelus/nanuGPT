import argparse
from typing import Callable, List, Type, Optional, Any, Union, Mapping, MutableMapping
from collections import UserDict
from typing import Sequence
from collections.abc import Mapping, MutableMapping
import os
from distutils.util import strtobool
import copy
import yaml
from datetime import datetime

_PREFIX_NODE = '_copy' # for copy node content command (must be dict)
_PREFIX_PATH = '_copy:' # for copy node value command (must be scaler)
_PREFIX_INHERIT = '_inherit' # for inherit node command, if true then inherit values else don't
_PREFIX_ENV = '_env' # command that sets the environment variables specified as key, values in dict
_PREFIX_TIME = '_time:' # command that evaluates to datetime string specified the format


def resolve_all(root_d:MutableMapping):
    _resolve_all(root_d, root_d, '/', set())

def _resolve_all(root_d:MutableMapping, cur:MutableMapping, cur_path:str, prev_paths:set):
    assert is_proper_path(cur_path)

    if cur_path in prev_paths:
        return # else we get in to infinite recursion
    prev_paths.add(cur_path)

    # if cur dict has '_copy' node with path in it
    child_path = cur.get(_PREFIX_NODE, None)
    if child_path and isinstance(child_path, str):
        # resolve this path to get source dict
        child_d = _resolve_path(root_d, _rel2full_path(cur_path, child_path), prev_paths)
        # we expect target path to point to dict so we can merge its keys
        if not isinstance(child_d, Mapping):
            raise RuntimeError(f'Path "{child_path}" should be dictionary but its instead "{child_d}"')
        # replace keys that have not been overriden
        _merge_source(child_d, cur)
        # remove command key
        del cur[_PREFIX_NODE]

    for k in cur.keys():
        # if this key needs path resolution, get target and replace the value
        rpath = _copy_command_val(cur[k])
        if rpath:
            cur[k] = _resolve_path(root_d,
                        _rel2full_path(_join_path(cur_path, k), rpath), prev_paths)

        time_fmt = _time_command_val(cur[k])
        if time_fmt is not None:
            time_fmt = '%Y%m%d-%H%M%S' if not time_fmt else time_fmt
            cur[k] = datetime.now().strftime(time_fmt)

        # if value is again dictionary, recurse on it
        if isinstance(cur[k], MutableMapping):
            _resolve_all(root_d, cur[k], _join_path(cur_path, k), prev_paths)

def _merge_source(source:Mapping, dest:MutableMapping)->None:
    # for anything that source has but dest doesn't, just do copy
    for sk in source:
        if sk not in dest:
            dest[sk] = source[sk]
        else:
            sv = source[sk]
            dv = dest[sk]

            # recursively merge child nodes
            if isinstance(sv, Mapping) and isinstance(dv, MutableMapping):
                _merge_source(source[sk], dest[sk])
            # else at least dest value is not dict and should not be overriden

def _copy_command_val(v:Any)->Optional[str]:
    """If the value is actually a path we need resolve then return that path or return None"""
    if isinstance(v, str) and v.startswith(_PREFIX_PATH):
        # we will almost always have space after _copy command
        return v[len(_PREFIX_PATH):].strip()
    return None

def _time_command_val(v:Any)->Optional[str]:
    """If the value is time command then return the specified format or return None"""
    if isinstance(v, str) and v.startswith(_PREFIX_TIME):
        # we will almost always have space after _copy command
        return v[len(_PREFIX_TIME):].strip()
    return None

def _join_path(path1:str, path2:str):
    mid = 1 if path1.endswith('/') else 0
    mid += 1 if path2.startswith('/') else 0

    # only 3 possibilities
    if mid==0:
        res = path1 + '/' + path2
    elif mid==1:
        res = path1 + path2
    else:
        res = path1[:-1] + path2

    return _norm_ended(res)

def _norm_ended(path:str)->str:
    if len(path) > 1 and path.endswith('/'):
        path = path[:-1]
    return path

def is_proper_path(path:str)->bool:
    return path.startswith('/') and (len(path)==1 or not path.endswith('/'))

def _rel2full_path(cwd:str, rel_path:str)->str:
    """Given current directory and path, we return abolute path. For example,
    cwd='/a/b/c' and rel_path='../d/e' should return '/a/b/d/e'. Note that rel_path
    can hold absolute path in which case it will start with '/'
    """
    assert len(cwd) > 0 and cwd.startswith('/'), 'cwd must be absolute path'

    rel_parts = rel_path.split('/')
    if rel_path.startswith('/'):
        cwd_parts = [] # rel_path is absolute path so ignore cwd
    else:
        cwd_parts = cwd.split('/')
    full_parts = cwd_parts + rel_parts

    final = []
    for i in range(len(full_parts)):
        part = full_parts[i].strip()
        if not part or part == '.': # remove blank strings and single dots
            continue
        if part == '..':
            if len(final):
                final.pop()
            else:
                raise RuntimeError(f'cannot create abs path for cwd={cwd} and rel_path={rel_path}')
        else:
            final.append(part)

    final = '/' + '/'.join(final)  # should work even when final is empty
    assert not '..' in final and is_proper_path(final) # make algo indeed worked
    return final


def _resolve_path(root_d:MutableMapping, path:str, prev_paths:set)->Any:
    """For given path returns value or node from root_d"""

    assert is_proper_path(path)

    # traverse path in root dict hierarchy
    cur_path = '/' # path at each iteration of for loop
    d = root_d
    for part in path.split('/'):
        if not part:
            continue # there will be blank vals at start

        # For each part, we need to be able find key in dict but some dics may not
        # be fully resolved yet. For last key, d will be either dict or other value.
        if isinstance(d, Mapping):
            # for this section, make sure everything is resolved
            # before we prob for the key
            _resolve_all(root_d, d, cur_path, prev_paths)

            if part in d:
                # "cd" into child node
                d = d[part]
                cur_path = _join_path(cur_path, part)
            else:
                raise RuntimeError(f'Path {path} could not be found in specified dictionary at "{part}"')
        else:
            raise KeyError(f'Path "{path}" cannot be resolved because "{cur_path}" is not a dictionary so "{part}" cannot exist in it')

    # last child is our answer
    rpath = _copy_command_val(d)
    if rpath:
        next_path = _rel2full_path(cur_path, rpath)
        if next_path == path:
            raise RuntimeError(f'Cannot resolve path "{path}" because it is circular reference')
        d = _resolve_path(root_d, next_path, prev_paths)
    return d

def deep_update(d:MutableMapping, u:Mapping, create_map:Callable[[],MutableMapping])\
        ->MutableMapping:
    # d is current state, u is new state
    # we start with empty, then merhe includes and then merge final file
    for k, v in u.items():
        if isinstance(v, Mapping):
            inherit = v.get(_PREFIX_INHERIT, True)

            # v is mapping so d[k] needs to be mapping to be able to merge
            # if k doesn't exist or set to None, create new mapping
            target = d.get(k, None)
            if target is None or not inherit:
                target = create_map()
                d[k] = target
            d[k] = deep_update(d[k], v, create_map)
        else:
            d[k] = v
    return d

def set_env_vars(root_d:MutableMapping):
    if _PREFIX_ENV in root_d:
        for k, v in root_d[_PREFIX_ENV].items():
            assert isinstance(v, str), f'Environment variable value in config key "{k}" must be string but got {v}'
            os.environ[k] = v

class Config(UserDict):
    def __init__(self, config_filepath:Optional[str]=None,
                 default_config_filepath:Optional[str]=None,
                 config_content:Optional[dict]=None,
                 app_desc:Optional[str]=None,
                 use_args=True,
                 first_arg_filename=True,
                 param_args: Sequence = [], run_commands=True) -> None:
        """Create config from specified yaml files and override args.

        Config is simply a dictionary of key, value which can form hirarchical
        config values. This class allows to load config from yaml and override
        any values using command line arguments using syntax
            --parent.child 42
        You can also use  `'__include__': another.yaml` to  specify base yaml.
        The base yaml is loaded first and the key-value pairs in the main yaml
        will override the ones in include file. Furthermore, you can also use
        `'_copy': 'path'` to copy values from one path to another inside yaml.
        For example,
            `'_copy': '/a/b/c'` will copy value of `/a/b/c` to current path.

        You can stop inheritance using `'_inherit': False`.

        Keyword Arguments:
            config_filepath {[str]} -- [Yaml file to load config from, could be names of files separated by semicolon which will be loaded in sequence oveeriding previous config] (default: {None})
            app_desc {[str]} -- [app description that will show up in --help] (default: {None})
            use_args {bool} -- [if true then command line parameters will override parameters from config files] (default: {False})
            param_args {Sequence} -- [parameters specified as ['--key1',val1,'--key2',val2,...] which will override parameters from config file.] (default: {[]})
            run_commands -- [if True then commands such as _copy, _env in yaml are executed]
            config_content -- if provided, this overrides content from the files, but not the command line args
            first_arg_filename -- if True then first arg is treated as config file name, else it is treated as normal arg
        """
        super(Config, self).__init__()

        self.args, self.extra_args = None, []

        if use_args:
            # let command line args specify/override config file
            parser = argparse.ArgumentParser(description=app_desc)
            # parser.add_argument("config_file", type=str, default=default_config_filepath,
            #     help='config filepath in yaml format, can be list separated by ;')
            self.args, self.extra_args = parser.parse_known_args()

            # expect first arg to be config filepath, if not supplied then use default
            if first_arg_filename and len(self.extra_args) > 0 and not self.extra_args[0].startswith('--'):
                config_filepath = self.extra_args.pop(0)
            else:
                config_filepath = default_config_filepath

        if config_filepath:
            for filepath in config_filepath.strip().split(';'):
                self._load_from_file(filepath.strip())
        if config_content is not None:
            deep_update(self, config_content, lambda: Config(run_commands=False, first_arg_filename=False))

        # Create a copy of ourselves and do the resolution over it.
        # This resolved_conf then can be used to search for overrides that
        # wouldn't have existed before resolution.
        resolved_conf = copy.deepcopy(self)
        if run_commands:
            resolve_all(resolved_conf)

        # Let's do final overrides from args
        self._update_from_args(param_args, resolved_conf)      # merge from params
        self._update_from_args(self.extra_args, resolved_conf) # merge from command line

        if run_commands:
            resolve_all(self)
            set_env_vars(self)

        self.config_filepath = config_filepath

    def _load_from_file(self, filepath:Optional[str])->None:
        # first we process includes and then we load source file
        # this allows source file to override included files
        if filepath:
            filepath = os.path.expanduser(os.path.expandvars(filepath))
            filepath = os.path.abspath(filepath)
            with open(filepath, 'r') as f:
                config_yaml = yaml.load(f, Loader=yaml.Loader)
            self._process_includes(config_yaml, filepath)
            deep_update(self, config_yaml, lambda: Config(run_commands=False, first_arg_filename=False))
            print('config loaded from: ', filepath)

    def _process_includes(self, config_yaml, filepath:str):
        if '__include__' in config_yaml:
            # include could be file name or array of file names to apply in sequence
            includes = config_yaml['__include__']
            if isinstance(includes, str):
                includes = [includes]
            assert isinstance(includes, List), "'__include__' value must be string or list"
            for include in includes:
                include_filepath = os.path.join(os.path.dirname(filepath), include)
                self._load_from_file(include_filepath)

    def _update_from_args(self, args:Sequence, resolved_section:'Config')->None:
        i = 0
        while i < len(args)-1:
            arg = args[i]
            if arg.startswith(("--")):
                path = arg[len("--"):].split('.')
                i += Config._update_section(self, path, args[i+1], resolved_section)
            else: # some other arg
                i += 1

    def to_dict(self)->dict:
        return deep_update({}, self, lambda: dict()) # type: ignore

    @staticmethod
    def _update_section(section:'Config', path:List[str], val:Any, resolved_section:'Config')->int:
        for p in range(len(path)-1):
            sub_path = path[p]
            if sub_path in resolved_section:
                resolved_section = resolved_section[sub_path]
                if not sub_path in section:
                    section[sub_path] = Config(run_commands=False, first_arg_filename=False)
                section = section[sub_path]
            else:
                return 1 # path not found, ignore this
        key = path[-1] # final leaf node value

        if key in resolved_section:
            original_val, original_type = None, None
            try:
                original_val = resolved_section[key]
                original_type = type(original_val)
                if original_type == bool: # bool('False') is True :(
                    original_type = lambda x: strtobool(x)==1
                section[key] = original_type(val)
            except Exception as e:
                raise KeyError(
                    f'The yaml key or command line argument "{key}" is likely not named correctly or value is of wrong data type. Error was occured when setting it to value "{val}".'
                    f'Originally it is set to {original_val} which is of type {original_type}.'
                    f'Original exception: {e}')
            return 2 # path was found, increment arg pointer by 2 as we use up val
        else:
            return 1 # path not found, ignore this

    def get_val(self, key, default_val):
        return super().get(key, default_val)

    @staticmethod
    def set_inst(instance:'Config')->None:
        global _config
        _config = instance

    @staticmethod
    def get_inst()->'Config':
        global _config
        return _config

