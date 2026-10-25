(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const state = { sample: '00020', condition: 'medium', samples: new Map() };
  let selection = 0;

  function error(message) {
    $('error').textContent = message;
    $('error').hidden = !message;
  }

  function busy(value) {
    $('loading').hidden = !value;
    $('comparison').setAttribute('aria-busy', String(value));
  }

  function setCondition(condition) {
    state.condition = condition;
    for (const button of $('conditions').children) button.setAttribute('aria-pressed', String(button.dataset.condition === condition));
  }

  function prediction(which, value) {
    const box = $(which + '-result');
    box.classList.toggle('correct', value.correct);
    box.classList.toggle('incorrect', !value.correct);
    box.querySelector('.verdict').textContent = value.correct ? '✓' : '×';
    $(which + '-label').textContent = value.label;
  }

  function preload(url) {
    const image = new Image();
    image.src = url;
    return image.decode();
  }

  async function showResult() {
    if (!state.samples.size) return;
    const current = ++selection;
    const sample = state.samples.get(state.sample);
    const result = sample?.results[state.condition];
    if (!result) { error('Không tìm thấy kết quả của ảnh này.'); return; }
    error(''); busy(true);
    try {
      await Promise.all([preload(result.images.before), preload(result.images.after)]);
      if (current !== selection) return;
      $('filename').textContent = sample.name;
      $('before-caption').textContent = result.condition === 'original'
        ? 'ResNet dự đoán · ảnh gốc (clean)'
        : 'ResNet dự đoán · ảnh thiếu sáng chưa tăng cường';
      for (const [id, source] of [['before-image', result.images.before], ['after-image', result.images.after],
                                  ['wipe-before-image', result.images.before], ['overlay-image', result.images.after]]) {
        $(id).src = source; $(id).hidden = false;
      }
      prediction('before', result.before);
      prediction('after', result.after);
      $('truth-label').textContent = result.truth.label;
      for (const button of $('samples').children) button.setAttribute('aria-pressed', String(button.dataset.sample === state.sample));
    } catch {
      if (current === selection) error('Không tải được ảnh. Hãy tải lại trang.');
    } finally {
      if (current === selection) busy(false);
    }
  }

  for (const button of $('conditions').children) button.addEventListener('click', () => {
    setCondition(button.dataset.condition); showResult();
  });
  $('compare-slider').addEventListener('input', event => $('comparison').style.setProperty('--split', event.target.value + '%'));

  async function start() {
    try {
      const response = await fetch('data/samples.json');
      if (!response.ok) throw new Error('Không tải được bộ kết quả demo.');
      const data = await response.json();
      if (data.schema_version !== 1 || data.mode !== 'precomputed' || data.samples.length !== 30) throw new Error('Bộ kết quả demo không hợp lệ.');
      for (const sample of data.samples) {
        state.samples.set(sample.id, sample);
        const button = document.createElement('button');
        button.type = 'button'; button.className = 'sample'; button.dataset.sample = sample.id;
        button.setAttribute('aria-label', sample.label + ' · ' + sample.name);
        button.setAttribute('aria-pressed', 'false'); button.title = sample.label + ' · ' + sample.name;
        const image = document.createElement('img'); image.src = sample.thumbnail; image.alt = '';
        button.append(image);
        button.addEventListener('click', () => { state.sample = sample.id; setCondition('medium'); showResult(); });
        $('samples').append(button);
      }
      if (!state.samples.has(state.sample)) state.sample = data.samples[0].id;
      await showResult();
    } catch (failure) {
      error(failure.message || 'Không tải được kết quả.'); busy(false);
      for (const button of $('conditions').children) button.disabled = true;
    }
  }
  start();
})();
