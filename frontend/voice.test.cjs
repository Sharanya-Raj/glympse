const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function setup(getUserMedia, api) {
  let stopped = 0, recorder, timer;
  const stream = { getTracks: () => [{ stop: () => stopped++ }] };
  class Recorder {
    static isTypeSupported(type) { return type.includes('webm'); }
    constructor() { recorder = this; this.state = 'inactive'; this.mimeType = 'audio/webm'; }
    start() { this.state = 'recording'; }
    stop() {
      this.state = 'inactive';
      queueMicrotask(() => {
        this.ondataavailable({ data: new Blob(['audio']) });
        this.onstop();
      });
    }
  }
  const context = { window: {}, navigator: { mediaDevices: { getUserMedia: getUserMedia || (async () => stream) } },
    MediaRecorder: Recorder, Blob, FormData, AbortController, Date,
    setInterval: callback => { timer = callback; return 1; }, clearInterval: () => { timer = null; } };
  context.window.MediaRecorder = Recorder;
  vm.runInNewContext(fs.readFileSync(__dirname + '/voice.js', 'utf8'), context);
  const requests = [];
  const voice = new context.window.MealDictation(api || { dictateMeal: async form => {
    requests.push(form); return { transcript: 'rice', items: [] };
  } }, () => {});
  return { voice, requests, stream, context, stopped: () => stopped, recorder: () => recorder, timer: () => timer };
}
const flush = () => new Promise(resolve => setImmediate(resolve));

test('stop uploads audio once and releases microphone for review', async () => {
  const h = setup();
  await h.voice.start();
  assert.equal(h.voice.state.phase, 'recording');
  h.voice.stop(); h.voice.stop();
  await flush();
  assert.equal(h.requests.length, 1);
  assert.equal(h.requests[0].get('audio').type, 'audio/webm');
  assert.equal(h.voice.state.phase, 'review');
  assert.equal(h.voice.state.result.transcript, 'rice');
  assert.equal(h.stopped(), 1);
  assert.equal(h.timer(), null);
});

test('cancel recording does not upload', async () => {
  const h = setup();
  await h.voice.start(); h.voice.cancel(); await flush();
  assert.equal(h.requests.length, 0);
  assert.equal(h.stopped(), 1);
  assert.equal(h.voice.state.phase, 'idle');
});

test('cancel while permission prompt is open releases late microphone stream', async () => {
  let resolve;
  const h = setup(() => new Promise(r => { resolve = r; }));
  const starting = h.voice.start();
  h.voice.cancel(); resolve(h.stream); await starting;
  assert.equal(h.stopped(), 1);
  assert.equal(h.recorder(), undefined);
});

test('permission rejection gives an actionable error', async () => {
  const h = setup(async () => { throw { name: 'NotAllowedError' }; });
  await h.voice.start();
  assert.match(h.voice.state.error, /denied/);
  assert.equal(h.requests.length, 0);
});

test('cancel processing aborts request and ignores late responses', async () => {
  let resolve, signal;
  const h = setup(null, { dictateMeal: (form, s) => { signal = s; return new Promise(r => { resolve = r; }); } });
  await h.voice.start(); h.voice.stop(); await flush();
  h.voice.cancel(); resolve({ transcript: 'stale', items: [] }); await flush();
  assert.equal(signal.aborted, true);
  assert.equal(h.voice.state.phase, 'idle');
  assert.equal(h.voice.state.result, null);
});

test('provider errors are shown and user can record again', async () => {
  const h = setup(null, { dictateMeal: async () => { throw new Error('Gemini key missing'); } });
  await h.voice.start(); h.voice.stop(); await flush();
  assert.match(h.voice.state.error, /Gemini key missing/);
  await h.voice.start();
  assert.equal(h.voice.state.phase, 'recording');
  h.voice.cancel();
});

test('two minute limit stops recording automatically', async () => {
  const h = setup();
  let now = 0;
  h.context.Date = { now: () => now };
  await h.voice.start(); now = 120000; h.timer()(); await flush();
  assert.equal(h.requests.length, 1);
  assert.equal(h.voice.state.phase, 'review');
});
