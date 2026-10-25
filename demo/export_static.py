"""Export the selected real paired predictions; production needs no Python or weights."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from .engine import ARCHIVE_SHA256, CHECKPOINT_SHA256, CLASS_NAMES, CONDITIONS, HERE, PIPELINE, Engine

PUBLIC = HERE / 'public'


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    print('Loading the verified seed-11 checkpoint for offline inference...', flush=True)
    engine = Engine()
    count = len(engine.samples)
    if count != 46:
        raise ValueError('Run python -m demo.prepare to prepare all 46 samples')
    conditions = ['original', *CONDITIONS]
    samples, files = [], {}
    matched = 0
    for index, (key, sample) in enumerate(engine.samples.items(), 1):
        if not re.fullmatch(r'\d{5}', key):
            raise ValueError('Invalid official sample ID')
        results = {}
        for condition in conditions:
            result = engine.analyze_sample(key, condition)
            if condition == 'original' and 'expected_original_before' in sample:
                for side in ['before', 'after']:
                    if result[side]['class_id'] != sample['expected_original_' + side]:
                        raise ValueError('Featured clean correction no longer reproduces')
            if condition == 'medium':
                for side in ['before', 'after']:
                    if result[side]['class_id'] != sample['expected_' + side]:
                        raise ValueError('Prediction differs from archived full-test result: ' + key)
                    matched += 1
            image_paths = {}
            for side in ['before', 'after']:
                relative = f'data/images/{key}/{condition}-{side}.png'
                path = PUBLIC / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(base64.b64decode(result['images'][side].split(',', 1)[1], validate=True))
                files[relative] = file_hash(path)
                image_paths[side] = relative
            results[condition] = dict(condition=condition, images=image_paths, before=result['before'],
                                      after=result['after'], truth=result['truth'])
        samples.append(dict(id=key, name=sample['filename'], class_id=sample['class_id'],
                            label=CLASS_NAMES[sample['class_id']], thumbnail=results['original']['images']['before'],
                            results=results))
        if index % 5 == 0:
            print(f'Exported {index}/{count} samples ({index * 4}/{count * 4} cases).', flush=True)
    data = dict(schema_version=1, mode='precomputed',
                model=dict(architecture='resnet18', seed=11, checkpoint_sha256=CHECKPOINT_SHA256),
                pipeline=PIPELINE, noise_seed=101, conditions=conditions, samples=samples)
    catalogue = PUBLIC / 'data/samples.json'
    catalogue_text = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    catalogue.write_text(catalogue_text + '\n', encoding='utf-8')
    files['data/samples.json'] = file_hash(catalogue)
    browser_catalogue = PUBLIC / 'data/samples.js'
    browser_catalogue.write_text('window.GTSRB_DEMO = ' + catalogue_text + ';\n', encoding='utf-8')
    files['data/samples.js'] = file_hash(browser_catalogue)
    source_names = ['demo/engine.py', 'demo/export_static.py', 'demo/prepare.py', 'demo/featured_example.json', 'src/common.py', 'src/model.py',
                    'src/degradation.py', 'src/enhancement.py', 'src/tools/processing_pipeline.py']
    evidence = dict(schema_version=1, mode='precomputed', generated_at=datetime.now(timezone.utc).isoformat(),
                    archive_sha256=ARCHIVE_SHA256, checkpoint_sha256=CHECKPOINT_SHA256,
                    prepared_samples_sha256=file_hash(engine.cache / 'samples.npz'),
                    sample_count=count, condition_count=4, case_count=count * 4, prediction_count=count * 8,
                    medium_predictions_matched_archive=matched, image_count=count * 8,
                    featured_clean_correction='00496', featured_original_predictions_verified=True,
                    inference='Actual CPU ResNet-18 inference on float32 arrays; no training or inference in the website',
                    display='PNG quantization for display only; model inputs remain float32',
                    source_sha256={name: file_hash(HERE.parent / name) for name in source_names}, files=files)
    (PUBLIC / 'data/provenance.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Complete: {count * 4} cases, {count * 8} predictions, {matched}/{count * 2} archived medium predictions matched.', flush=True)
    print(f'Static files: {PUBLIC}', flush=True)


if __name__ == '__main__':
    main()
