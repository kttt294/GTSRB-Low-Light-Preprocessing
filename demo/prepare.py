"""Prepare the frozen classifier and 46 official ROI examples; no training."""
import csv
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from src.common import digest
from src.data import crop_box
from src.degradation import degrade
from .engine import ARCHIVE_SHA256, CACHE, CHECKPOINT_SHA256, CONDITIONS, HERE, PIPELINE

ROOT = HERE.parent


def prepare(archive, data_root, cache=CACHE):
    audit = json.loads((ROOT / 'report/evidence/final/visual_cases_audit.json').read_text(encoding='utf-8'))
    if digest(archive) != ARCHIVE_SHA256 or audit['result_archive_sha256'] != ARCHIVE_SHA256:
        raise ValueError('Results archive differs from the verified report')
    for record in audit['image_sources']:
        path = data_root / record['url'].rsplit('/', 1)[-1]
        if digest(path) != record['sha256']:
            raise ValueError('Official GTSRB archive checksum mismatch: ' + path.name)
    with zipfile.ZipFile(data_root / 'GTSRB_Final_Test_GT.zip') as ground_truth:
        member = next(name for name in ground_truth.namelist() if name.endswith('GT-final_test.csv'))
        annotations = sorted(csv.DictReader(io.StringIO(ground_truth.read(member).decode('utf-8-sig')), delimiter=';'),
                             key=lambda row: row['Filename'])
    with zipfile.ZipFile(archive) as results:
        checkpoint = results.read('05_training/cnn_11.pt')
        if hashlib.sha256(checkpoint).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError('Checkpoint checksum mismatch')
        protocol = json.loads(results.read('08_processing_extension/fulltest/protocol.json'))
        if protocol['registry'][audit['method']] != PIPELINE or protocol['config']['conditions'] != CONDITIONS:
            raise ValueError('Demo configuration differs from the executed protocol')
        cases = list(audit['cases'])
        represented = {int(case['class_id']) for case in cases}
        remaining = sorted(set(range(43)) - represented)
        # Retain the six paper examples, then cover 24 other classes across the ID range.
        # Select the first official test ID per class, before reading predictions.
        selected_classes = [remaining[index] for index in np.linspace(0, len(remaining) - 1, 24).round().astype(int)]
        allowed_ids = set(protocol['sample_ids'])
        for class_id in selected_classes:
            row = next(row for row in annotations if int(row['ClassId']) == class_id
                       and 'gtsrb/test/' + Path(row['Filename']).stem in allowed_ids)
            box = crop_box([row[key] for key in ['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2']],
                           int(row['Width']), int(row['Height']),
                           padding=protocol['config']['crop']['padding'],
                           inclusive=protocol['config']['crop']['roi_end_inclusive'])
            cases.append(dict(sample_id='gtsrb/test/' + Path(row['Filename']).stem,
                              class_id=class_id, padded_crop_box=box))
        if len(cases) != 30 or len({int(case['class_id']) for case in cases}) != 30:
            raise ValueError('Expected 30 distinct samples and classes')
        featured = json.loads((HERE / 'featured_example.json').read_text(encoding='utf-8'))
        featured_name = featured['sample_id'].rsplit('/', 1)[-1] + '.ppm'
        official = next(row for row in annotations if row['Filename'] == featured_name)
        official_box = crop_box([official[key] for key in ['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2']],
                                int(official['Width']), int(official['Height']),
                                padding=protocol['config']['crop']['padding'],
                                inclusive=protocol['config']['crop']['roi_end_inclusive'])
        if (featured['sample_id'] not in allowed_ids or int(official['ClassId']) != featured['class_id']
                or list(official_box) != featured['padded_crop_box'] or featured['checkpoint_sha256'] != CHECKPOINT_SHA256):
            raise ValueError('Featured sample does not match official label, crop or checkpoint')
        cases.insert(0, featured)
        if len({case['sample_id'] for case in cases}) != 31:
            raise ValueError('Featured sample duplicates an existing example')
        represented = {int(case['class_id']) for case in cases}
        extra_classes = sorted(set(range(43)) - represented) + [0, 21, 42]
        if len(extra_classes) != 15:
            raise ValueError('Expected 12 missing classes and three additional examples')
        for class_id in extra_classes:
            used = {case['sample_id'] for case in cases}
            row = next(row for row in annotations if int(row['ClassId']) == class_id
                       and 'gtsrb/test/' + Path(row['Filename']).stem in allowed_ids
                       and 'gtsrb/test/' + Path(row['Filename']).stem not in used)
            box = crop_box([row[key] for key in ['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2']],
                           int(row['Width']), int(row['Height']),
                           padding=protocol['config']['crop']['padding'],
                           inclusive=protocol['config']['crop']['roi_end_inclusive'])
            cases.append(dict(sample_id='gtsrb/test/' + Path(row['Filename']).stem,
                              class_id=class_id, padded_crop_box=box))
        if len(cases) != 46 or len({case['sample_id'] for case in cases}) != 46:
            raise ValueError('Expected 46 unique examples')
        predictions = {}
        wanted = {case['sample_id'] for case in cases}
        with results.open('08_processing_extension/fulltest/predictions.csv') as binary:
            for row in csv.DictReader(io.TextIOWrapper(binary, encoding='utf-8', newline='')):
                if (row['sample_id'] in wanted and row['condition'] == 'medium'
                        and row['noise_seed'] == '101' and row['train_seed'] == '11'
                        and row['method'] in {'identity', audit['method']}):
                    key = row['sample_id'], row['method']
                    if key in predictions:
                        raise ValueError('Duplicate archived prediction')
                    predictions[key] = row
                    if len(predictions) == 2 * len(cases):
                        break
        if len(predictions) != 2 * len(cases):
            raise ValueError('Missing archived predictions for selected samples')
    arrays, samples = {}, []
    with zipfile.ZipFile(data_root / 'GTSRB_Final_Test_Images.zip') as raw:
        for case in cases:
            sample_id = case['sample_id']
            key = sample_id.rsplit('/', 1)[-1]
            filename = key + '.ppm'
            source = raw.read('GTSRB/Final_Test/Images/' + filename)
            source_sha = hashlib.sha256(source).hexdigest()
            if 'source_image_sha256' in case and source_sha != case['source_image_sha256']:
                raise ValueError('Official image differs from audited source: ' + key)
            with Image.open(io.BytesIO(source)) as image:
                clean = np.asarray(image.convert('RGB').crop(case['padded_crop_box']).resize(
                    (64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255
            dark = degrade(clean, **CONDITIONS['medium'], noise_seed=101, sample_id=sample_id, condition='medium')
            for name, image in [('clean', clean), ('dark', dark)]:
                if name + '_float32_sha256' in case and hashlib.sha256(image.tobytes()).hexdigest() != case[name + '_float32_sha256']:
                    raise ValueError('Reconstructed sample differs from audited pixels: ' + key)
            before = predictions[sample_id, 'identity']
            after = predictions[sample_id, audit['method']]
            if int(before['class_id']) != int(case['class_id']) or int(after['class_id']) != int(case['class_id']):
                raise ValueError('Official label differs from prediction record')
            arrays[key] = clean
            sample = dict(id=key, sample_id=sample_id, filename=filename,
                                class_id=int(case['class_id']),
                                source_image_sha256=source_sha,
                                expected_before=int(before['prediction']), expected_after=int(after['prediction']))
            if 'original_before' in case:
                sample.update(expected_original_before=case['original_before'], expected_original_after=case['original_after'])
            samples.append(sample)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / 'cnn_11.pt').write_bytes(checkpoint)
    np.savez_compressed(cache / 'samples.npz', **arrays)
    manifest = dict(archive_sha256=ARCHIVE_SHA256, checkpoint_sha256=CHECKPOINT_SHA256,
                    samples_sha256=digest(cache / 'samples.npz'), pipeline=PIPELINE, conditions=CONDITIONS,
                    samples=samples, noise_seed=101, train_seed=11,
                    selection='Featured clean correction, original 30 examples, then first unused ID for 12 missing classes and classes 0/21/42; covers all 43 classes; illustrative only')
    (cache / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Ready: frozen ResNet-18 seed 11 and 46 GTSRB samples covering 43 classes; 00496 first.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'outputs/kaggle-20261025-v2/gtsrb-notebook-results.zip')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/raw/gtsrb')
    args = parser.parse_args()
    try:
        prepare(args.archive, args.data_root)
    except FileNotFoundError as exc:
        parser.exit(1, f'Missing input: {exc.filename}. See demo/README.md.\n')


if __name__ == '__main__':
    main()
