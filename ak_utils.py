from IPython.core.display import display, HTML
import json
from pprint import pprint
from copy import deepcopy
import pandas as pd
import pickle
from glob import glob
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

USE_PLOTLY = False

def html_format(obj, indent = 1):
    """HTML formatter specific for the `DeepDict` library type and for specific HTML markup"""
    if isinstance(obj, list):
        htmls = []
        for k in obj:
            htmls.append(html_format(k))
        return '['+ ", ".join(htmls)+']'

    if isinstance(obj, dict):
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
            l = k.split(key_separator)
            if len(l) == 1:
                d[k] = v
            else:
                p = l[0]
                if p in d.keys():
                    cur_val = d[p]
                    assert type(cur_val) == dict, f'Keys that have a value cannot be used also as part of a cascading path: {p} from {flat_dict} has value {d[p]}'
                else:
                    cur_val = {}
                d[p] = cascade_dict({key_separator.join(l[1:]): v}, cur_output=cur_val)
    return d
    
 
class AutoKerasSearchSpace(object):
    """Read and represent the search space of an AutoKeras project"""
    
    def __init__(self, project_name, dir='.'):
        """Initialize with the 'project_name' used for the AutoKeras task"""
        self.dir = dir
        self.project_name = project_name
        self.block_type_config_dict = {}
        self.defaults = {'generic': {}}
        
    def load(self):
        """Load the search space definition from the oracle.json file of the project"""
        try:
            fname = f'{self.dir}/{self.project_name}/oracle.json'
            with open(fname) as f:
                jsonconfig = json.load(f)
            self._read_config(jsonconfig)
        except:
            print(f'Failed to read {fname}')    
            
    def _read_config(self, jsonconfig):
        """Read space configuration as set of tuples (key, default, possible values, conditions).
        
        The object classes are defined in `kerastuner/engine/hyperparameters.py`.
        Not dealt with below are Float and Fixed classes, which aren't used in practice.
        """
        
        space = jsonconfig['hyperparameters']['space']
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
        
    def _config_per_block_type(self) -> dict:
        """Convert list of tuples from `_read_config` to a dictionary where the keys are block_types. 
        
        If there is no block_type, 'generic' is used.
        """
        
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

    def display_configuration(self) -> dict:
        """Display the configuration"""
        
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
            
    def get_best_model(self):
        """Get the best model of the trial run. """
        
        try:
            fname = f'{self.dir}/{self.project_name}/best_model'
            model = tf.keras.models.load_model(fname)
            return model
        except Exception as e:
            print(f'Failed to load model at {fname}', e)   
            
    def get_trial_data(self, trial_id):
        return self._trials[trial_id]
    
    def trial_config(self, trial_id) -> dict:
        """Hyperparameter values for trial"""
        filename = f'{self.dir}/{self.project_name}/trial_{trial_id}/trial.json'
        with open(filename, 'r') as f:
            j_data = json.load(f)
        b_values = j_data['hyperparameters']['values']
        return deepcopy(b_values)
    
    def display_trial_configuration(self, trial_id) -> dict:
        """Display the configuration of a trial.
        
        Values are colored:
        * red for default, 
        * blue for actual value, 
        * green if actual value is the same as the default value.
        """
        
        d = self._config_per_block_type()
        t_config = self.trial_config(trial_id)
        out_d = {}
        for block_type, values in d.items():
            for key, default, possible, _ in values:     
                replace = {}
                if default in possible:
                    ind = possible.index(default)
                    replace[ind] = 'red'
                if key in t_config.keys() and t_config[key] in possible:
                    ind = possible.index(t_config[key])
                    replace[ind] = 'blue'
                if key in t_config.keys() and t_config[key] == default:
                    replace[possible.index(default)] = 'green'
                for ind, col in replace.items():
                    possible[ind] = f'<text style="color:{col}">{possible[ind]}</text>'
                if not block_type in out_d:
                    out_d[block_type] = {}
                out_d[block_type][key] = possible
        assert len(d) == len(out_d), f"dictionary length changed from {len(d)} to {len(out_d)}"
        return out_d
    
    def count_parameter_values(self, block_type=None) -> int:
        """Count the number of values for hyperparameters. If `block_type` is set, only for that block_type."""
        if block_type:
            return sum([len(x[2]) for x in self._config_per_block_type()[block_type]])
        else:
            p = [x[2] for x in self._config]
            print(p)
            return sum([len(x[2]) for x in self._config])
        
    def hyperparameters_per_block_type(self) -> pd.DataFrame:
        """The number of hyperparameters for each `block_type`.
        
        Returns
        -------
          DataFrame:
            index: block_type
            count: number of parameters specific for block_type
            count_plus_generic: number of aprameters specific for block_type plus the number of common parameters
        """
        d = {}
        block_types = [b for b in self._config_per_block_type().keys()]
        generic_count = 0
        if 'generic' in block_types:
            generic_count = self.count_parameter_values(block_type='generic')
        for b in block_types:
            if b == 'generic' and len(block_types) > 1:
                # Skip the `generic` block_type, unless it is the only type
                continue
            else:
                c = self.count_parameter_values(block_type=b)
                d[b] = [c, c + generic_count]
        return pd.DataFrame.from_dict(d).T.reindex().rename(columns={0: 'count', 1: 'count_plus_generic'})
    

class AutoKerasTrials(AutoKerasSearchSpace):
    def __init__(self, project_name, dir='.'):
        super(AutoKerasTrials, self).__init__(project_name, dir)
        self._trial_dict = ldict()
        self._trials = {}
            
    def _load_trial_data_json(self, filename, metrics_only=False, metric='val_accuracy', block_type='reference') -> dict:
        with open(filename, 'r') as f:
            j_data = json.load(f)
        
        d = {}
        if not metrics_only:
            # the block_type is always in the second layer and is by itself a unique key
            block_type = 'generic' # Default of `generic` to deal with AutoModels

            b_values = j_data['hyperparameters']['values']
            for k, v in b_values.items():
                d[k] = v
                # identify the block_type (so we have it without a prefix for the type of task)
                if k.endswith('block_type'):
                    block_type = v
            d['block_type'] = block_type
            d['trial_id'] = j_data['trial_id']        
            d['best_step'] = j_data['best_step']
        else:
            d['block_type'] = block_type
        
        observations = j_data['metrics']['metrics'][metric]['observations']
        d['max_val_accuracy'] = max(list(map(lambda x: x['value'][0], observations)))
        
        metrics_keys = j_data['metrics']['metrics'].keys()
        if metric in j_data['metrics']['metrics'].keys():
            data = j_data['metrics']['metrics'][metric]
            d['metric_direction'] = data['direction']
            for m in data['observations']:
                d[f"step_{m['step']}"] = m['value'][0]
        else:
            print(f"No metric {metric} in data in {filename}. Skipping metric data.")
        return d

    def _load_trial_data(self, filename, metric='val_accuracy'):
        d = self._load_trial_data_json(filename, metric=metric)
        
        parent_dir = Path(filename).parent
        lst = glob(f'{parent_dir}/checkpoints/*/checkpoint.data-00000-of-00001')
        if lst:
            d['size'] = os.stat(lst[0]).st_size
            d['mtime'] = datetime.fromtimestamp(os.stat(lst[0]).st_mtime)
        else:
            # In `testdata` we have no checkpoints
            assert str(parent_dir).startswith('testdata'), f'Checkpoints missing for {filename}'
            
        self._trials[d['trial_id']] = d
        self._trial_dict.add_dict(d)
                
    def load_trials(self, metric='val_accuracy', with_reference=False):
        """Load data on all trials and return as DataFrame"""
        self._trial_dict = ldict()
        self._trials = {}
        trial_lst = glob(f'{self.dir}/{self.project_name}/trial_*/trial.json')
        trial_lst.sort(key=os.path.getctime)
        for fname in trial_lst: 
            #try:
            self._load_trial_data(fname, metric=metric)
            #except Exception as a:
            #    print(f"Error reading file {fname}: {a}")
        
        if with_reference:
            ref = ImageClassifierReference()
            ref_dict = self._load_trial_data_json(ref.get_filename(), metrics_only=True)
            ref_dict['trial_id'] = 'REFERENCE'
            ref_dict['size'] = ref.get_model_file_size()
            ref_dict['image_block_1/block_type'] = 'xception'
            ref_dict['best_step'] = -1
            
            # Remove epochs done in reference training in excess of the actual trials
            epochs = self.count_epochs()
            
            to_remove = []
            allowed_steps = [f'step_{x}' for x in range(epochs)]
            for k in ref_dict.keys():
                if k.startswith('step_') and k not in allowed_steps:
                    to_remove.append(k)
            for k in to_remove:
                ref_dict.pop(k)
            
            self._trials['REFERENCE'] = ref_dict
            self._trial_dict.add_dict(ref_dict)

        df = pd.DataFrame(self._trial_dict).sort_values(by='max_val_accuracy', ascending=False).astype({'best_step': 'int32'})
        
        return df
    
    def count_trials(self):
        return self._trial_dict.depth()
    
    def count_epochs(self):
        """(Maximum) number of epochs in trials
        
        Effectively count the number of keys identifying steps.
        """
        
        return len([x for x in self._trial_dict.keys() if x.startswith('step')])
    
    def summary(self):
        block_types = list(self._config_per_block_type().keys())
        if len(block_types) > 1 and 'generic' in block_types:
            block_types.remove('generic')

        n = len(block_types)
        plural_postfix = 's' if n > 0 else ''        
        return f'{n} architecture type{plural_postfix}: {block_types}, {self.count_trials()} trials, {self.count_epochs()} epochs'
    
    
class Analyzer(object):
    """Display results in Jupyter notebook"""
    
    def __init__(self, project_name=None, dir='.'):
        assert project_name, "'project_name' must be set"
        analyzer = AutoKerasTrials(project_name, dir=dir)
        analyzer.load()
        analyzer.load_trials()
        self.analyzer = analyzer
        self.next_trial_counter = 0
    
    def display_search_space(self, architecture_type=None):
        for bt, conf in self.analyzer.display_configuration().items():
            if bt == 'generic' or not architecture_type or architecture_type == bt:
                display(HTML(f'<h3>{bt}</h3>'))
                display(HTML(html_format(cascade_dict(conf))))
                
    def plot_trial_run(self, trial_id=None, auto_trial=False):
        """validation_accuracy per epoch
        
        If no arguments are given, the best trial for each architecture type will be given. If a reference
        trial exists, that will also be displayed.

        parameters:
          trial_id: plot only for this `trial_id`
          auto_trial: every time the Jupyter cell is executed again, a new trial will be displayed
        """
        
        # !pip install statsmodels
        with_reference = not auto_trial and not trial_id
        
        # FIXME: for now, 'reference' is not implemented, so we set `with_reference` to False
        with_reference = False
        df = self.dataframe_trials(with_reference=with_reference) # normally with reference, but not for auto_trial
        step_cols = [m for m in df.columns if m.startswith('step_')]
        epochs = self.analyzer.count_epochs()
        
        if auto_trial:  
            trial_id = df.iloc[self.next_trial_counter]['trial_id']
            self.next_trial_counter += 1
            if self.next_trial_counter > len(step_cols):
                self.next_trial_counter = 0
        
        if trial_id:
            d = self.analyzer.get_trial_data(trial_id)
            display(HTML(f"<ul><li>Trial id: {d['trial_id']}</li><li>Architecture: {d['block_type']}</li><li>max val accuracy: {d['max_val_accuracy']}</li></ul>"))
            steps = df.iloc[self.next_trial_counter][step_cols].to_frame(name='val_accuracy').reset_index().rename(columns={'index': 'step'})
     
            # We use the index instead of he 'step_n' column to create the diagram
            if USE_PLOTLY:
                fig = px.scatter(steps.reset_index(), x='index', y='val_accuracy', trendline="lowess")
                fig.update_layout(title='Validation accuracy per epoch')
                fig.show()
            else:
                # Down to matplotlib
                x = steps.reset_index()['index']
                y = steps['val_accuracy']
                plt.scatter(x, y)
                plt.show()
        else:
            block_types = df['block_type'].unique()
            if USE_PLOTLY:
                fig = go.Figure()

            for bt in block_types:
                crit = df['block_type'] == bt

                line_df = df[crit].head(1)
                cur_df = line_df[step_cols].T.reset_index().reset_index()
                if len(cur_df.columns) < 3:
                    print('Bad data for dataframe, skipping', bt)
                    display(cur_df)
                    continue

                cur_df.columns = ['index', 'step', 'val_accuracy']
    
                linedict = None
                if bt == 'reference':
                    linedict = dict(dash='dot')
                if USE_PLOTLY:
                    fig.add_trace(go.Scatter(x=cur_df['index'],
                             y=cur_df['val_accuracy'],
                             name=bt,
                             line=linedict,
                             visible=True))

            if USE_PLOTLY:
                fig.update_layout(title='Validation accuracy per epoch for trail resulting in max accuracy for each type')
                fig.show()
            else:
                plt.show()
            
    def plot_trials_over_time(self):
        df = self.dataframe_trials(with_reference=False)
        if USE_PLOTLY:
            fig = px.scatter(df.sort_values(by='mtime', ascending=True), x="mtime", y="max_val_accuracy", color='block_type', title="Model searches - validation accuracy", hover_data=['trial_id'])
            fig.show()
        else:
            # FIXME
            ndf = df.sort_values(by='mtime', ascending=True)
            x = ndf['mtime']
            y = ndf['max_val_accuracy']
            def bt_to_color(bt):
                if bt == 'xception':
                    return 'red'
                elif bt == 'resnet':
                    return 'blue'
                elif bt == 'vanilla':
                    return 'green'
                else:
                    return 'black'
            ndf['color'] = ndf['block_type'].apply(bt_to_color)
            c = ndf['color']
            plt.scatter(x, y, c=c)
            plt.show()
        
    def plot_count_per_architecture_type(self):
        return self.analyzer.hyperparameters_per_block_type().plot.pie(y='count', autopct='%1.f%%', title='Number of parameters per block_type')
    
    def plot_best_step(self, block_type=None):
        df = self.dataframe_trials(with_reference=False)
        if block_type == 'all':
            block_types = list(self.analyzer._config_per_block_type().keys())
            if len(block_types) > 1 and 'generic' in block_types:
                block_types.remove('generic')
            print('block types are: ', block_types)
            for bt in block_types:
                condition = df['block_type'] == bt
                best_step_df = df[condition]
                display(best_step_df.hist('best_step')) #, title='Frequency of best step')
        else:
            condition = True
            if block_type:
                condition = df['block_type'] == block_type
                best_step_df = df[condition]
            else:
                best_step_df = df
            return best_step_df.hist('best_step') #, title='Frequency of best step')
  
    def dataframe_trials(self, metric='val_accuracy', with_reference=False):
        return self.analyzer.load_trials(metric=metric, with_reference=with_reference)
        
    def summary(self):
        return self.analyzer.summary()
        
    
class ldict(dict):
    """Maintain dictionary as with equal length lists as values.
    
    A pandas `DataFrame` can be instantiated by a dictionary where the values are lists of equal length.
    `ldict` builds up such a dictionary.
    
    Example:
      d = ldict()
      d.add_dict({'a': 0, 'b': 7})
      d.add_dict({'a': 2, 'c': 4})
      d.add_dict({'a': 3, 'b': 8, 'c': 5})
      
      results in:
        { 'a': [0, 2, 3], 'b': [7, None, 8], 'c': [None, 4, 5]}
    """
    def __init__(self, *args, **kw):
        super(ldict,self).__init__(*args, **kw)
        self.elemlength = 0
        
    def add_dict(self, d):
        """Add the items of the argument dictionary d keeping the values of all keys as equal length lists.
        
        The values in the argument dictionary are added to the values for the key in the object.
        """
        for k, v in d.items():
            if not k in self.keys():
                self.__setitem__(k, [None] * self.elemlength)
            self.__getitem__(k).append(v)
                
        self.elemlength += 1

        for k in self.keys():
            if len(self.__getitem__(k)) < self.elemlength:
                self.__getitem__(k).append(None)
                
    def depth(self):
        """Number of values per key"""
        if self.__len__ == 0:
            return 0
        else:
            return len(list(self.values())[0])

        
class ReferenceTrial(object):
    """A reference trial for a task."""
    def __init__(self, task_name, dir="."):
        self.task_name = task_name
        self.dir = dir
        self.data = None
    
    def execute(self):
        print(f'executing {self.task_name}')
    
    def save(self):
        print(f'saving {self.task_name}')
    
    def get_filename(self):
        return f'{self.task_name.lower()}_reference.json'
    
    def get_model_file_size(self):
        lst = glob('reference_xception/variables/variables.data-00000-*')
        if lst:
            return os.stat(lst[0]).st_size
        else:
            return -1
    
    
class TextClassifierReference(ReferenceTrial):
    def __init__(self, dir="."):
        super(TextClassifierReference, self).__init__('TextClassifier', dir=dir)
        
    def execute(self):
        print(f'executing {self.task_name}')
        

class ImageClassifierReference(ReferenceTrial):
    def __init__(self, dir="."):
        super(ImageClassifierReference, self).__init__('ImageClassifier', dir=dir)
        
    def execute(self):
        print(f'executing {self.task_name}')
        
    def load(self):
        with open(self.get_filename(), 'r') as f:
            self.data = json.load(f)
        return self.data
    
    def save(self):
        with open('resnet50-history.pkl', 'rb') as f:
            newhist = pickle.load(f)
        
        d = {'metrics': {
            'loss': {
                'direction': 'min', 
                'observations': []}, 
            'accuracy': {
                'direction': 'max',
                'observations': []}, 
            'val_loss': {
                'direction': 'min',
                'observations': []}, 
            'val_accuracy': {
                'direction': 'max',
                'observations': []}
        }}
        for k, l in newhist.items():
            for i, v in enumerate(l):
                d['metrics'][k]['observations'].append({'value': [v], 'step': i})
                
        with open(self.get_filename(), 'w') as f:
            json.dump({'metrics': d}, f)
            
            
                
if __name__ == '__main__':
    from pprint import pprint
    ref = ImageClassifierReference()
    ref.execute()
    ref.save()
    data = ref.load()
    pprint(data)
    
    project_name = 'image_classifier_greedy'
    akss = AutoKerasTrials(project_name)
    akss.load()
    df = akss.load_trials(metric='val_accuracy', with_reference=True)
    print(df[['block_type', 'max_val_accuracy']])
    
    #m = akss.get_best_model()
    #if m:
    #    m.summary()
    #else:
    #    print('no best model')
    akss = AutoKerasTrials("image_classifier_greedy")
    akss.load()
    trial_id = '38c96975f74e72507f06fd3bd02fdc26'
    p = akss.display_trial_configuration(trial_id)