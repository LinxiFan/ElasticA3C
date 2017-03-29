"""
Data and config serialization.
"""
import functools
import json
import os
from ..common import AttributeDict

"json_parse: string -> dict"
json_parse = json.loads

"json_str: dict -> string"
json_str = functools.partial(json.dumps, indent=4)

        
class JsonWriter(AttributeDict):
    """
    Load and write json files like an AttributeDict
    """
    def __init__(self, json_file, *args, **kwargs):
        AttributeDict.__init__(self, *args, **kwargs)
        self._json_file = os.path.expanduser(json_file)
        if os.path.exists(self._json_file):
            self.update(json_load(self._json_file))
    
    def to_dict(self):
        # avoid writing internal `_json_file` to the output file
        d = dict(self)
        d.pop('_json_file')
        return d
    
    def save(self):
        json_save(self._json_file, self.to_dict())


def json_load(filepath):
    with open(filepath, 'r') as fp:
        return json.load(fp)


def json_save(filepath, dat):
    with open(filepath, 'w') as fp:
        json.dump(dat, fp, indent=4, sort_keys=True)