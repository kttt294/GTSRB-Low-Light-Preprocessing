"""Offline paired inference used only to export the static demo."""
import base64
from functools import lru_cache
import io
import json
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np
from PIL import Image
import torch

from src.common import digest
from src.degradation import degrade
from src.model import image_tensor, make_cnn
from src.tools.processing_pipeline import Pipeline

HERE = Path(__file__).resolve().parent
CACHE = HERE / '.cache'
ARCHIVE_SHA256 = 'c09dfbeafeacfff98e1e21429d5078aa6dfde2ce2a5a774c130dc966a4cf1e6f'
CHECKPOINT_SHA256 = '1eb63e6c674f592c066eded4b02923c093019264a2be1db757f1015936ad7164'
PIPELINE = {'steps': [
    {'kind': 'denoise', 'method': 'gaussian', 'params': {'kernel': 5, 'sigma': 0.8}},
    {'kind': 'enhance', 'method': 'gamma', 'params': {'gamma': 0.8}},
]}
CONDITIONS = {
    'mild': {'gamma_dark': 1.5, 'poisson_peak': 120, 'gaussian_sigma': 0.005},
    'medium': {'gamma_dark': 2.2, 'poisson_peak': 60, 'gaussian_sigma': 0.01},
    'severe': {'gamma_dark': 3.0, 'poisson_peak': 30, 'gaussian_sigma': 0.02},
}
CLASS_NAMES = (
    '20 km/h', '30 km/h', '50 km/h', '60 km/h', '70 km/h', '80 km/h',
    'Hết giới hạn 80 km/h', '100 km/h', '120 km/h', 'Cấm vượt',
    'Cấm xe tải vượt', 'Ưu tiên tại giao lộ', 'Đường ưu tiên', 'Nhường đường',
    'Dừng lại', 'Cấm xe cơ giới', 'Cấm xe tải', 'Cấm đi ngược chiều',
    'Nguy hiểm khác', 'Ngoặt nguy hiểm trái', 'Ngoặt nguy hiểm phải',
    'Nhiều chỗ ngoặt nguy hiểm', 'Đường gồ ghề', 'Đường trơn',
    'Đường hẹp bên phải', 'Công trường', 'Tín hiệu đèn giao thông',
    'Người đi bộ', 'Trẻ em', 'Xe đạp', 'Băng tuyết', 'Động vật hoang dã',
    'Hết mọi lệnh cấm', 'Rẽ phải', 'Rẽ trái', 'Đi thẳng',
    'Đi thẳng hoặc rẽ phải', 'Đi thẳng hoặc rẽ trái', 'Đi bên phải',
    'Đi bên trái', 'Đi vòng xuyến', 'Hết cấm vượt', 'Hết cấm xe tải vượt',
)


def png_data(image):
    output = io.BytesIO()
    Image.fromarray(np.rint(image * 255).clip(0, 255).astype(np.uint8)).save(output, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode('ascii')


class Engine:
    def __init__(self, cache=CACHE):
        self.cache = Path(cache)
        metadata = json.loads((self.cache / 'manifest.json').read_text(encoding='utf-8'))
        if metadata['archive_sha256'] != ARCHIVE_SHA256 or metadata['pipeline'] != PIPELINE or metadata['conditions'] != CONDITIONS:
            raise ValueError('Prepared demo does not match the verified experiment')
        checkpoint_path = self.cache / 'cnn_11.pt'
        if digest(checkpoint_path) != CHECKPOINT_SHA256:
            raise ValueError('Classifier checkpoint checksum mismatch')
        sample_path = self.cache / 'samples.npz'
        if digest(sample_path) != metadata['samples_sha256']:
            raise ValueError('Sample cache checksum mismatch')
        self.samples = {sample['id']: sample for sample in metadata['samples']}
        with np.load(sample_path, allow_pickle=False) as arrays:
            self.clean = {key: arrays[key].copy() for key in self.samples}
        torch.set_num_threads(4)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint['seed'] != 11 or checkpoint['cnn_config']['architecture'] != 'resnet18':
            raise ValueError('Expected the frozen ResNet-18, seed 11')
        self.model = make_cnn(checkpoint['cnn_config'], pretrained=False).eval()
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.pipeline = Pipeline(PIPELINE, torch.device('cpu'))
        self.lock = Lock()

    def analyze(self, clean, sample_id, filename, condition, truth=None):
        if condition not in {'original', *CONDITIONS}:
            raise ValueError('Mức thiếu sáng không hợp lệ.')
        before = clean.copy() if condition == 'original' else degrade(
            clean, **CONDITIONS[condition], noise_seed=101, sample_id=sample_id, condition=condition)
        started = perf_counter()
        with self.lock, torch.inference_mode():
            after = self.pipeline(before, sample_id)
            batch = torch.stack([image_tensor(before), image_tensor(after)])
            probabilities = self.model(batch).softmax(dim=1).numpy()
        elapsed = (perf_counter() - started) * 1000
        def prediction(probs):
            class_id = int(probs.argmax())
            return dict(class_id=class_id, label=CLASS_NAMES[class_id], confidence=float(probs[class_id]),
                        correct=None if truth is None else class_id == truth)
        return dict(filename=filename, condition=condition,
                    images={'before': png_data(before), 'after': png_data(after)},
                    before=prediction(probabilities[0]), after=prediction(probabilities[1]),
                    truth=None if truth is None else dict(class_id=truth, label=CLASS_NAMES[truth]),
                    model={'architecture': 'resnet18', 'seed': 11, 'checkpoint_sha256': CHECKPOINT_SHA256},
                    processing_ms=round(elapsed, 1))

    @lru_cache(maxsize=120)
    def analyze_sample(self, sample_key, condition):
        sample = self.samples[sample_key]
        return self.analyze(self.clean[sample_key], sample['sample_id'], sample['filename'], condition, sample['class_id'])

