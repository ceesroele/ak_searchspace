from IPython.core.display import display, HTML
from collections.abc import MutableMapping
import json
from pprint import pprint
from copy import deepcopy
import deprecation

def html_format(obj, indent = 1):
    """HTML formatter specific for the `DeepDict` library type and for specific HTML markup"""
    if isinstance(obj, list):
        htmls = []
        for k in obj:
            htmls.append(html_format(k))
        return '['+ ", ".join(htmls)+']'

    if isinstance(obj, DeepDict) or isinstance(obj, dict):
        htmls = []
        for k,v in obj.items():
            htmls.append("<span style='font-style: italic; color: #888'>%s</span>: %s" % (k,html_format(v,indent+1)))

        return '{<div style="margin-left: %dem">%s</div>}' % (indent, ',<br>'.join(htmls))

    if type(obj) == str and obj.startswith("'<text") and obj.endswith("/text>'"):
        obj = obj[1:-1]
        
    return str(obj)

def cascade_dict(flat_dict, cur_output={}, key_separator='/', skip_first=False):
    """Split dictionary by keys to create a cascaded dictionary, 
    e.g. {'foo/bar': 10} becomes {'foo': {'bar': 10}}"""
    assert type(flat_dict) == dict
    d = {}
    if (skip_first):
        for k, v in flat_dict.items():
            if type(v) == dict:
                d[k] = cascade_dict(v)
            else:
                d[k] = v
    else:
        d = deepcopy(cur_output)
        for k, v in flat_dict.items():
            assert type(k) == str, f'cascade_dict(..) can only deal with strings as keys, got {k}'
            l = k.split('/')
            if len(l) == 1:
                d[k] = v
            else:
                p = l[0]
                if p in d.keys():
                    cur_val = d[p]
                    assert type(cur_val) == dict, f'Keys that have a value cannot be used also as part of a cascading path: {p} from {flat_dict} has value {d[p]}'
                else:
                    cur_val = {}
                d[p] = cascade_dict({"/".join(l[1:]): v}, cur_output=cur_val)
    return d
    
    
class AutoKerasSearchSpace(object):
    """Read and represent the search space of an AutoKeras project"""
    def __init__(self, project_name):
        """Initialize with the 'project_name' used for the AutoKeras task"""
        self.project_name = project_name
        self.config = None
        self.parameters = DeepDict()
        self.parameters_dict = {}
        self.conditions = DeepDict()
        self.block_types = []
        self.block_type_config = DeepDict()
        self.block_type_config_dict = {}
        self.defaults = {'generic': {}}
        
    def load(self):
        """Load the search space definition from the oracle.json file of the project"""
        try:
            fname = f'./{self.project_name}/oracle.json'
            with open(fname) as f:
                self.config = json.load(f)
            self._parse()
            self._read_config()
        except:
            print(f'Failed to read {fname}')    
            
    def _parse(self):
        for p in self.config['hyperparameters']['space']:
            d = self.parameters
            e = self.parameters_dict
            val = None
            
            # We use that in practice:
            # 1. There is always only one condition
            # 2. This condition is always one of the block_types
            block_type = None
            if 'conditions' in p['config'] and p['config']['conditions']:
                for k in p['config']['conditions']:
                    c = k['config']
                    name = c['name']
                    assert len(c['values']) == 1, f"Expected single value for condition, but got: {c['values']}"
                    val = c['values'][0]
                    if name.endswith('block_type'):
                        block_type = val
                        if not block_type in self.block_type_config.keys():
                            self.block_type_config[block_type] = DeepDict()
                            self.block_type_config_dict[block_type] = {}
                            
                        if not block_type in self.defaults.keys():
                            self.defaults[block_type] = {}
                        d = self.block_type_config[block_type]
                        e = self.block_type_config_dict[block_type]
                    else:
                        self.conditions[name] = val
            
            default_key = block_type
            if default_key is None:
                default_key = 'generic'
            if p['class_name'] == 'Choice':
                if p['config']['name'].endswith('/block_type'):
                    self.block_types = p['config']['values']
                d[p['config']['name']] = p['config']['values']
                self.defaults[default_key][p['config']['name']] = p['config']['default']
            elif p['class_name'] == 'Boolean':
                d[p['config']['name']] = [True, False]
                self.defaults[default_key][p['config']['name']] = p['config']['default']
            elif p['class_name'] == 'Int':
                # We convert the combo of min_value, max_value, and steps to a list of integer values
                min_value = p['config']['min_value']
                max_value = p['config']['max_value']
                steps = p['config']['step']
                d[p['config']['name']] = [min_value, max_value]
            else:
                print(f"No handing for {p['class_name']}", p)        

    def _read_config(self):
        """Read space configuration as set of tuples (key, default, possible values, conditions).
        The object classes are defined in kerastuner/engine/hyperparameters.py.
        Not dealt with below are Float and Fixed classes, as I didn't see them in practice.
        """
        space = self.config['hyperparameters']['space']
        defaults = [(x['config']['name'], x['config']['default']) for x in space]
        values = []
        for x in space:
            if x['class_name'] == 'Boolean':
                values.append([True, False])
            elif x['class_name'] == 'Int':
                min_value = x['config']['min_value']
                max_value = x['config']['max_value']
                step = x['config']['step']
                v = [x for x in range(min_value, max_value+step, step)]
                values.append(v)
            else:
                values.append(x['config']['values'])
        
        conditions = [x['config']['conditions'] if 'conditions' in x['config'].keys() else [] for x in space]
        self._config = [(x[0], x[1], y, z) for x, y, z in zip(defaults, values, conditions)]
        
    def _config_per_block_type(self):
        """Convert list of tuples from _read_config to a dictionary where the keys are block_types. If there is no block_type, 'generic' is used."""
        d = {}
        for name, default, values, conditions in self._config:
            block_type = 'generic'
            new_conditions = {}
            for c in conditions:
                if c['config']['name'].endswith('block_type'):
                    block_type = c['config']['values'][0]  # values is a list, normally with one element
                else:
                    new_conditions[c['config']['name']] = c['config']['values'][0]
            if block_type in d.keys():
                v = d[block_type]
            else:
                v = []
            v.append((name, default, values, new_conditions))
            d[block_type] = v
        return d

    def display_configuration(self):
        """Display the configuration"""
        self._read_config()
        d = self._config_per_block_type()
        out_d = {}
        for block_type, values in d.items():
            for key, default, possible, _ in values:       
                colored_possibles = self._color(default, possible, color="red")
                if not block_type in out_d:
                    out_d[block_type] = {}
                out_d[block_type][key] = colored_possibles
        assert len(d) == len(out_d), f"dictionary length changed from {len(d)} to {len(out_d)}"
        return out_d
    
    def _color(self, item, lst, color='blue'):
        """If item is in the list, mark it with (HTML) color"""
        res = []
        for it in lst:
            if item == it:
                res.append(f'<text style="color:{color}">{it}</text>')
            else:
                res.append(it)
        return res
        
    def __str__(self):
        return "\n".join([f'{x} -> {y}' for x, y in self.parameters.items()])
    
    def trial_config(self, trial_id):
        filename = f'./{self.project_name}/trial_{trial_id}/trial.json'
        with open(filename, 'r') as f:
            j_data = json.load(f)
        b_values = j_data['hyperparameters']['values']
        return deepcopy(b_values)
    
    def display_trial_configuration(self, trial_id):
        """Display the configuration of a trial. 
        Values are colored red for default, blue for actual value, 
        green if actual value is the same as the default value.
        """
        self._read_config()
        d = self._config_per_block_type()
        t_config = self.trial_config(trial_id)
        out_d = {}
        for block_type, values in d.items():
            for key, default, possible, _ in values:     
                replace = {}
                if default in possible:
                    ind = possible.index(default)
                    replace[ind] = 'red'
                if t_config[key] in possible:
                    ind = possible.index(t_config[key])
                    replace[ind] = 'blue'
                if t_config[key] == default:
                    replace[possible.index(default)] = 'green'
                for ind, col in replace.items():
                    possible[ind] = f'<text style="color:{col}">{possible[ind]}</text>'
                if not block_type in out_d:
                    out_d[block_type] = {}
                out_d[block_type][key] = possible
        assert len(d) == len(out_d), f"dictionary length changed from {len(d)} to {len(out_d)}"
        return out_d

    ## FIXME: I want to remove this
    def get_defaults_for_block_type(self, block_type):
        d = DeepDict()
        d[block_type] = deepcopy(self.block_type_config[block_type])
        d['generic'] = deepcopy(self.parameters)
        return d
            
    ## FIXME: I want to remove this
    def XXXfill_in_defaults(self, orig_dict, default_dict):
        #assert isinstance(orig_dict, DeepDict), "Wrong type for orig_dict, must be DeepDict"
        #assert type(default_dict) == dict, "Wrong type for default-dict, must be dict"
        d = deepcopy(orig_dict)
        for k, v in default_dict.items():
            r = d[k]
            if type(r) == list:
                t = []
                for p in r:
                    if p == v:
                        t.append('<text style="color:blue">'+str(v)+'</text>')
                    else:
                        t.append(p)
                d[k] = t
            else:
                d[k] = '<text style="color:blue">'+str(v)+'</text>'
        return d
                    

# FIXME: I want to remove this
class DeepDict(MutableMapping):
    """Dictionary in which complex keys are broken up into separate keys serving to cascade the dictionary.
    Example: DeepDict['foo/bar'] = 10 gets the structure {'foo': {'bar': 10}}"""
    def __init__(self, *args, **kwargs):
        """Use the object dict"""
        self.__dict__.update(*args, **kwargs)
    
    def __setitem__(self, key, value):
        """if key is a string, split it into a list if it contains a '/'
        if key is a list, cascade into deeper dictionaries
        otherwise act as with a normal dictionary"""
        if type(key) == str:
            l = key.split('/')
            if len(l) > 1:
                key = l
            
        if type(key) == list:
            if len(key) == 0:
                pass
            elif len(key) == 1:
                self.__dict__[key[0]] = value
            else:
                k = key[0]
                rest = key[1:]
                
                d = None
                if k in self.__dict__.keys():
                    d = self.__dict__[k]
                else:
                    d = DeepDict()        
                    self.__dict__[k] = d
                d[rest] = value
        else:
            self.__dict__[key] = value
    
    def __getitem__(self, key):
        if type(key) == str:
            l = key.split('/')
            if len(l) > 1:
                l_key = l
            else:
                l_key = [key]
        elif type(key) == list:
            l_key = key
        else:
            return self.__dict__[key]
            
        res = self.__dict__
        for k in l_key:
            try:
                res = res[k]
            except KeyError as e:
                print(f"orig key: {key}, subkey: {k}, dict keys: {str(res.keys())}")
                raise e
        return res
    
    def __delitem__(self, key):
        del self.__dict__[key]
    
    def __iter__(self):
        return iter(self.__dict__)
    
    def __len__(self):
        return len(self.__dict__)
    
    # The final two methods aren't required, but nice for demo purposes:
    def __str__(self):
        """returns simple dict representation of the mapping"""
        return str(self.__dict__)
    
    def __repr__(self):
        return repr(self.__dict__)
    
    def todict(self):
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, DeepDict):
                d[k] = v.todict()
            else:
                d[k] = v
        return d
    
    def flatten(self, toflattendict=None):
        if toflattendict is None:
            toflattendict = self.__dict__
        d = {}
        for k, v in toflattendict.items():
            if isinstance(v, DeepDict):
                p = self.flatten(v.__dict__)
                for k1, v1 in p.items():
                    d[f'{k}/{k1}'] = v1
            else:
                d[k] = v
        return d
    
    def count_values(self, key=None):
        """Count the number of values"""
        if key:
            return self.__getitem__(key).count_values()
        else:
            count = 0
            for k, v in self.__dict__.items():
                if isinstance(v, DeepDict):
                    count += v.count_values()
                elif type(v) == list:
                    count += len(v)
                else:
                    count += 1
            return count
    

class ldict(dict):
    """Maintain dictionary as with equal length lists as values, e.g. 
    { 'a': [0, 2, 3], 'b': [7, None, 8], 'c': [None, 4, 5]}"""
    def __init__(self, *args, **kw):
        super(ldict,self).__init__(*args, **kw)
        self.elemlength = 0
        
    def add_dict(self, d):
        """Add the items of the argument dictionary d to the object. 
        The values in the argument dictionary are added to the values for the key in the object."""
        for k, v in d.items():
            if not k in self.keys():
                self.__setitem__(k, [None] * self.elemlength)
            self.__getitem__(k).append(v)
                
        self.elemlength += 1

        for k in self.keys():
            if len(self.__getitem__(k)) < self.elemlength:
                self.__getitem__(k).append(None)
                
                
# ------------ Tests -----------------------
             
def test_cascade_dict():
    input1 = {'foo/bar/fie': 10, 'foo/doo': 15}
    output1 = {'foo': {'bar': {'fie': 10}, 'doo': 15}}
    
    d = cascade_dict(input1)
    assert d == output1
    
    input2 = {'skip': {'foo/bar/fie': 10, 'foo/doo': 15}}
    output2 = {'skip': {'foo': {'bar': {'fie': 10}, 'doo': 15}}}
    
    d = cascade_dict(input2, skip_first=True)
    assert d == output2

    input3 = {'skip': {'foo/bar/fie': 10, 'foo/doo': 15}}
    output3 = {'skip': {'foo/bar/fie': 10, 'foo/doo': 15}}
    
    d = cascade_dict(input3, skip_first=False)
    assert d == output3

# ------------- DeepDict tests -------------
              
def _deepdict_setup():
    d = DeepDict()
    d['fie'] = 7
    d['foo/bar'] = [1,2,3]
    d['foo/fiefo'] = {'a': '1'}
    return d

def test_deepdict_structure():
    d = _deepdict_setup()
    assert d['foo']['bar'] == [1,2,3]
    assert d['fie'] == 7
    assert d['fie'] != 'bar'
    assert d['foo']['fiefo']['a'] == '1'
    
def test_deepdict_count_values():
    d = _deepdict_setup()
    assert d.count_values() == 5

def test_deepdict_keyerror():
    d = _deepdict_setup()
    has_keyerror = False
    try:
        d['no_key_for_this']
    except KeyError as e:
        has_keyerror = True
    assert has_keyerror == True
    
def test_deepdict_flatten():
    d = _deepdict_setup()
    x = d.flatten()
    assert type(x) == dict
    assert 'foo/bar' in x.keys()
    assert x['foo/bar'] == d['foo/bar']
    
    
# ------------- ldict tests -----------------------
def test_ldict():
    d1 = {'a': 1, 'b': 2}
    d2 = {'b': 5, 'c': 6}
    ld = ldict()
    ld.add_dict(d1)
    ld.add_dict(d2)
    
    assert ld == {'a': [1, None], 'b': [2, 5], 'c': [None, 6]}
    

# ------------- AutoKerasSearchSpace tests --------

def _autokerassearchspace_setup():
    defaults = {'foo/bar': 10, 'fie': 20}

    
def test_autokerassearchspace_fill_in_defaults():
    pass
    
    
if __name__ == '__main__':
    #print(test_autokerassearchspace_trial_to_deepdict())
    #input = {'foo/bar/fie': 10, 'foo/doo': 15}
    #print('original input: ', input)
    #res = cascade_dict(input)
    #print('outcome = ',res)
    akss = AutoKerasSearchSpace('image_classifier_greedy')
    akss.load()
    from pprint import pprint
    #pprint(cascade_dict(akss.display_configuration(),skip_first=True))
    
    for bt, conf in akss.display_configuration().items():
        print(f'<h3>{bt}</h3>')
        #display(HTML(html_format(cascade_dict(conf))))
        pprint(cascade_dict(conf))
              
    #print(cascade_dict({'skip': {'foo/bar/fie': 10, 'foo/doo': 15}}, skip_first=True))
    print(cascade_dict( {'skip': {'foo/bar/fie': 10, 'foo/doo': 15}}))