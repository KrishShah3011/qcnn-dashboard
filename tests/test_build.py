import json
import numpy as np
from build_data import to_js, clean, find_boundaries


def test_to_js_nan(tmp_path):
    to_js('demo', {'a': np.array([1.0, np.nan, np.inf]), 'b': np.float32(2.5)}, out_dir=str(tmp_path))
    txt = (tmp_path / 'demo.js').read_text(encoding='utf-8')
    assert txt.startswith('window.QCNN = window.QCNN || {};\nQCNN.demo = ')
    obj = json.loads(txt.split('QCNN.demo = ', 1)[1].rstrip().rstrip(';'))
    assert obj == {'a': [1.0, None, None], 'b': 2.5}


def test_find_boundaries_synthetic():
    h1 = np.array([0.0, 0.5])
    h2 = np.round(np.arange(-1.6, 1.6001, 0.1), 2)
    e = np.stack([-np.sqrt((h2 - 0.4) ** 2 + 0.01) - np.sqrt((h2 + 1.2) ** 2 + 0.01)] * 2, axis=1)
    b = find_boundaries(h1, h2, e)
    assert len(b['upper']) == 2 and abs(b['upper'][0][1] - 0.4) < 0.03
    assert abs(b['lower'][0][1] + 1.2) < 0.03
