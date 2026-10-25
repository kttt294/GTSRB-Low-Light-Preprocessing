"""Validate the published static bundle without importing ML dependencies."""
import hashlib
import json
from pathlib import Path
import struct
import unittest

PUBLIC = Path(__file__).resolve().parent / 'public'


class StaticBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((PUBLIC / 'data/samples.json').read_text(encoding='utf-8'))
        cls.audit = json.loads((PUBLIC / 'data/provenance.json').read_text(encoding='utf-8'))

    def test_complete_thirty_class_four_condition_matrix(self):
        self.assertEqual(self.data['mode'], 'precomputed')
        samples = self.data['samples']
        self.assertEqual(len(samples), 30)
        self.assertEqual(len({s['id'] for s in samples}), 30)
        self.assertEqual(len({s['class_id'] for s in samples}), 30)
        for sample in samples:
            self.assertEqual(set(sample['results']), {'original', 'mild', 'medium', 'severe'})

    def test_predictions_and_correctness_agree_with_official_label(self):
        for sample in self.data['samples']:
            for condition, result in sample['results'].items():
                self.assertEqual(result['condition'], condition)
                self.assertEqual(result['truth']['class_id'], sample['class_id'])
                self.assertEqual(result['truth']['label'], sample['label'])
                for side in ['before', 'after']:
                    prediction = result[side]
                    self.assertIn(prediction['class_id'], range(43))
                    self.assertGreaterEqual(prediction['confidence'], 1 / 43)
                    self.assertLessEqual(prediction['confidence'], 1)
                    self.assertEqual(prediction['correct'], prediction['class_id'] == sample['class_id'])

    def test_all_assets_match_hashes_and_stay_inside_public(self):
        self.assertEqual(self.audit['medium_predictions_matched_archive'], 60)
        self.assertEqual(self.audit['case_count'], 120)
        self.assertEqual(self.audit['prediction_count'], 240)
        self.assertEqual(self.audit['image_count'], 240)
        self.assertEqual(len(self.audit['files']), 241)
        for name, expected in self.audit['files'].items():
            path = (PUBLIC / name).resolve()
            self.assertTrue(path.is_relative_to(PUBLIC.resolve()))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected, name)

    def test_images_exist_and_are_64_pixel_rgb_pngs(self):
        images = set()
        for sample in self.data['samples']:
            self.assertTrue((PUBLIC / sample['thumbnail']).is_file())
            for result in sample['results'].values():
                for name in result['images'].values():
                    pixels = (PUBLIC / name).read_bytes()
                    self.assertEqual(pixels[:8], b'\x89PNG\r\n\x1a\n')
                    self.assertEqual(struct.unpack('>II', pixels[16:24]), (64, 64))
                    self.assertEqual(pixels[25], 2)
                    images.add(name)
        self.assertEqual(len(images), 240)

    def test_site_has_no_uploads_or_server_api(self):
        html = (PUBLIC / 'index.html').read_text(encoding='utf-8')
        script = (PUBLIC / 'app.js').read_text(encoding='utf-8')
        for text in [html, script]:
            self.assertNotIn('/api/', text)
            self.assertNotIn('FormData', text)
            self.assertNotIn('image-file', text)
            self.assertNotIn('open-image', text)
        self.assertIn('class="demo-note">Demo</span>', html)
        self.assertIn("fetch('data/samples.json')", script)
        self.assertFalse(any(PUBLIC.rglob('*.pt')))
        self.assertFalse(any(PUBLIC.rglob('*.py')))

    def test_original_six_cases_still_match_report(self):
        report = json.loads((PUBLIC.parent.parent / 'report/evidence/final/visual_cases_audit.json').read_text(encoding='utf-8'))
        samples = {sample['id']: sample for sample in self.data['samples']}
        for case in report['cases']:
            result = samples[case['sample_id'].rsplit('/', 1)[-1]]['results']['medium']
            self.assertEqual(result['before']['class_id'], int(case['before']))
            self.assertEqual(result['after']['class_id'], int(case['after']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
