from ak_utils import ldict, cascade_dict, AutoKerasSearchSpace, AutoKerasTrials

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
    return AutoKerasTrials('text_classifier_greedy', dir='testdata')
    
def test_autokerassearchspace___init__():
    _autokerassearchspace_setup()
    
def test_autokerassearchspace_load():
    akss = _autokerassearchspace_setup()
    akss.load()
    assert len(akss._config) == 39
    assert akss._config[-1] == ('text_block_1/bert_block_1/max_seq_len', 128, [128, 256, 512], [{'class_name': 'Parent', 'config': {'name': 'text_block_1/block_type', 'values': ['bert']}}])

def test_autokerassearchspace_load_trials():
    akss = _autokerassearchspace_setup()
    akss.load()
    akss.load_trials()
    
def test_autokerassearchspace_summary():
    akss = _autokerassearchspace_setup()
    akss.load()
    akss.load_trials()
    a = akss.summary()
    assert a == "3 architecture types: ['vanilla', 'transformer', 'bert'], 4 trials, 3 epochs", f'Got: {a}'

def test_autokerassearchspace_fill_in_defaults():
    pass