"""Reconstruct six fixed test examples and verify saved CNN predictions before plotting."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common import digest
from src.data import ARCHIVES, BASE_URL, crop_box
from src.degradation import degrade
from src.model import make_cnn, image_tensor
from src.tools.processing_pipeline import Pipeline

NAMES = {
    3: 'Giới hạn 60 km/h', 4: 'Giới hạn 70 km/h', 5: 'Giới hạn 80 km/h',
    7: 'Giới hạn 100 km/h', 12: 'Đường ưu tiên', 19: 'Ngoặt nguy hiểm trái',
    22: 'Đường gồ ghề', 23: 'Đường trơn', 38: 'Đi bên phải',
    40: 'Đi vòng xuyến',
}
CATEGORIES = {'helped': 'Được giúp', 'harmed': 'Bị làm sai', 'still_wrong': 'Vẫn sai'}
METHOD = 'pre__gaussian__gamma'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/raw/gtsrb')
    args = parser.parse_args()
    evidence = ROOT / 'report/evidence/final'
    audit = json.loads((evidence / 'audit.json').read_text(encoding='utf-8'))
    if digest(args.archive) != audit['archive_sha256']:
        raise ValueError('Results archive does not match the completed audit')
    with (evidence / 'representative_cases.csv').open(encoding='utf-8', newline='') as stream:
        cases = list(csv.DictReader(stream))
    if len(cases) != 6 or [r['category'] for r in cases] != ['helped'] * 2 + ['harmed'] * 2 + ['still_wrong'] * 2:
        raise ValueError('Expected the original six fixed representative cases')
    for name in ['GTSRB_Final_Test_Images.zip', 'GTSRB_Final_Test_GT.zip']:
        with (args.data_root / name).open('rb') as stream:
            if hashlib.file_digest(stream, 'md5').hexdigest() != ARCHIVES[name]:
                raise ValueError('Official data archive checksum mismatch: ' + name)
    with zipfile.ZipFile(args.data_root / 'GTSRB_Final_Test_GT.zip') as z:
        member = next(n for n in z.namelist() if n.endswith('GT-final_test.csv'))
        annotations = {r['Filename']: r for r in csv.DictReader(io.StringIO(z.read(member).decode('utf-8-sig')), delimiter=';')}
    records = []
    images = []
    with zipfile.ZipFile(args.archive) as z, zipfile.ZipFile(args.data_root / 'GTSRB_Final_Test_Images.zip') as raw:
        checksums = json.loads(z.read('SHA256.json'))
        # The prior audit covers all files; verify the exact members consumed here again.
        def verified(name):
            data = z.read(name)
            expected = checksums.get(name)
            if isinstance(expected, dict):
                expected = expected['sha256']
            if sha(data) != expected:
                raise ValueError('Result member checksum mismatch: ' + name)
            return data
        protocol = json.loads(verified('08_processing_extension/fulltest/protocol.json'))
        checkpoint_bytes = verified('05_training/cnn_11.pt')
        expected_checkpoint = next(r['sha256'] for r in protocol['checkpoints'] if r['seed'] == 11)
        if sha(checkpoint_bytes) != expected_checkpoint:
            raise ValueError('Wrong classifier checkpoint')
        with zipfile.ZipFile(io.BytesIO(verified('06_reproduce/project_source.zip'))) as source:
            for name in ['src/common.py', 'src/data.py', 'src/degradation.py', 'src/model.py',
                         'src/enhancement.py', 'src/tools/processing_pipeline.py']:
                bundled = source.read(name).decode('utf-8').replace('\r\n', '\n')
                local = (ROOT / name).read_text(encoding='utf-8').replace('\r\n', '\n')
                if bundled != local:
                    raise ValueError('Local processing source differs from executed source: ' + name)
        selected = {}
        with z.open('08_processing_extension/fulltest/predictions.csv') as binary:
            wanted = {r['sample_id'] for r in cases}
            for r in csv.DictReader(io.TextIOWrapper(binary, encoding='utf-8', newline='')):
                if (r['sample_id'] in wanted and r['condition'] == 'medium' and r['noise_seed'] == '101'
                        and r['train_seed'] == '11' and r['method'] in {'identity', METHOD}):
                    key = (r['sample_id'], r['method'])
                    if key in selected:
                        raise ValueError('Duplicate prediction')
                    selected[key] = r
        if len(selected) != 12:
            raise ValueError('Missing paired predictions')
        pipeline = Pipeline(protocol['registry'][METHOD], torch.device('cpu'))
        for case in cases:
            sid = case['sample_id']
            filename = sid.rsplit('/', 1)[-1] + '.ppm'
            annotation = annotations[filename]
            label = int(annotation['ClassId'])
            if label != int(case['class_id']) or sid not in protocol['sample_ids']:
                raise ValueError('Official label or sample ID mismatch')
            image_bytes = raw.read('GTSRB/Final_Test/Images/' + filename)
            original = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            if original.size != (int(annotation['Width']), int(annotation['Height'])):
                raise ValueError('Official dimensions disagree')
            box = crop_box([annotation[k] for k in ['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2']],
                           *original.size, padding=protocol['config']['crop']['padding'],
                           inclusive=protocol['config']['crop']['roi_end_inclusive'])
            clean = np.asarray(original.crop(box).resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255
            dark = degrade(clean, **protocol['config']['conditions']['medium'], noise_seed=101, sample_id=sid, condition='medium')
            enhanced = pipeline(dark, sid)
            images.append((clean, dark, enhanced))
            record = dict(case, official_class_id=label, official_roi=[int(annotation[k]) for k in ['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2']],
                          padded_crop_box=box, source_image_sha256=sha(image_bytes),
                          clean_float32_sha256=sha(clean.tobytes()), dark_float32_sha256=sha(dark.tobytes()),
                          enhanced_float32_sha256=sha(enhanced.tobytes()))
            for method, field in [('identity', 'before'), (METHOD, 'after')]:
                row = selected[sid, method]
                if int(row['class_id']) != label or row['prediction'] != case[field]:
                    raise ValueError('Representative case differs from the audited prediction')
                record[field + '_archived_prediction'] = int(row['prediction'])
                record[field + '_archived_true_probability'] = float(row['true_probability'])
            records.append(record)
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True)
    if checkpoint['manifest_sha256'] != protocol['manifest_sha256'] or checkpoint['seed'] != 11:
        raise ValueError('Checkpoint provenance mismatch')
    model = make_cnn(checkpoint['cnn_config'], pretrained=False).eval()
    model.load_state_dict(checkpoint['state_dict'])
    torch.set_num_threads(4)
    batch = torch.stack([image_tensor(im) for triple in images for im in triple[1:]])
    with torch.inference_mode():
        probabilities = model(batch).softmax(1).numpy()
    for i, record in enumerate(records):
        for j, field in enumerate(['before', 'after']):
            probs = probabilities[2 * i + j]
            prediction = int(probs.argmax())
            if prediction != record[field + '_archived_prediction']:
                raise ValueError('Reconstructed image does not reproduce archived prediction: ' + record['sample_id'])
            record[field + '_local_prediction'] = prediction
            truth = int(record['class_id'])
            record[field + '_local_true_probability'] = float(probs[truth])
            record[field + '_true_probability_absolute_delta'] = abs(float(probs[truth]) - record[field + '_archived_true_probability'])
            if record[field + '_true_probability_absolute_delta'] > 1e-4:
                raise ValueError('Reconstructed image probability differs beyond CPU/GPU tolerance')
    print('Verified official labels and all 12 reconstructed-image predictions.', flush=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig = plt.figure(figsize=(9.2, 7.8), facecolor='white')
    grid = fig.add_gridspec(6, 4, width_ratios=[1, 1, 1, 2.6], hspace=.14, wspace=.12,
                           left=.01, right=.99, bottom=.005, top=.95)
    for i, (record, triple) in enumerate(zip(records, images)):
        for j, image in enumerate(triple):
            ax = fig.add_subplot(grid[i, j])
            ax.imshow(image, interpolation='nearest', vmin=0, vmax=1)
            ax.axis('off')
            if i == 0:
                ax.set_title(['Ảnh gốc', 'Ảnh tối', 'Gaussian → Gamma'][j], fontsize=11.5, pad=8)
        ax = fig.add_subplot(grid[i, 3]); ax.axis('off')
        truth, before, after = int(record['class_id']), int(record['before']), int(record['after'])
        text = (f"{CATEGORIES[record['category']]} · {record['sample_id'].rsplit('/', 1)[-1]}\n"
                f"Thật: {truth} · {NAMES[truth]}\n"
                f"Tối: {before} · {NAMES[before]}\n"
                f"Xử lý: {after} · {NAMES[after]}")
        ax.text(0, .5, text, va='center', fontsize=12, linespacing=1.4)
    output = ROOT / 'report/latex/images/visual_error_cases.png'
    fig.savefig(output, dpi=240, facecolor='white'); plt.close(fig)
    metadata = dict(status='official_labels_and_all_12_predictions_verified', result_archive_sha256=audit['archive_sha256'],
                    prediction_member='08_processing_extension/fulltest/predictions.csv', condition='medium', noise_seed=101,
                    train_seed=11, checkpoint_sha256=expected_checkpoint, method=METHOD, pipeline=protocol['registry'][METHOD],
                    selection='Two smallest sample IDs per helped/harmed/still_wrong group, unchanged from representative_cases.csv',
                    image_sources=[dict(url=BASE_URL + name, md5=ARCHIVES[name], sha256=digest(args.data_root / name))
                                   for name in ['GTSRB_Final_Test_Images.zip', 'GTSRB_Final_Test_GT.zip']],
                    display='Same RGB [0,1] scale; nearest-neighbor magnification, no per-image contrast normalization',
                    class_names='Vietnamese descriptions of GTSRB IDs; IDs remain the canonical labels',
                    inference_device='cpu', source_comparison='Exact UTF-8 source text after CRLF/LF normalization',
                    true_probability_tolerance=1e-4, packages={'numpy': np.__version__, 'torch': torch.__version__},
                    figure_sha256=digest(output), script_sha256=digest(__file__), cases=records)
    (evidence / 'visual_cases_audit.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Wrote', output, flush=True)


if __name__ == '__main__':
    main()
