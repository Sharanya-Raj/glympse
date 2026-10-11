/* Browser microphone -> authenticated meal dictation. Audio stays in memory. */
(function () {
  'use strict';
  class MealDictation {
    constructor(api, onChange) {
      this.api = api;
      this.onChange = onChange;
      this.state = { phase: 'idle', seconds: 0, error: '', result: null };
      this.generation = 0;
    }

    update(values) {
      Object.assign(this.state, values);
      this.onChange();
    }

    release() {
      clearInterval(this.timer);
      this.timer = null;
      if (this.stream) this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }

    cancel() {
      this.generation++;
      if (this.request) this.request.abort();
      this.request = null;
      if (this.recorder && this.recorder.state !== 'inactive') this.recorder.stop();
      this.recorder = null;
      this.release();
      this.update({ phase: 'idle', seconds: 0, error: '', result: null });
    }

    fail(message) {
      this.cancel();
      this.update({ phase: 'error', error: message });
    }

    async start() {
      if (['requesting', 'recording', 'processing'].includes(this.state.phase)) return;
      this.cancel();
      const generation = this.generation;
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        return this.fail('Microphone recording needs a supported browser on HTTPS or localhost.');
      }
      this.update({ phase: 'requesting' });
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        if (generation !== this.generation) {
          stream.getTracks().forEach(track => track.stop());
          return;
        }
        this.stream = stream;
        const mimeType = ['audio/webm;codecs=opus', 'audio/mp4', 'audio/ogg;codecs=opus', 'audio/webm']
          .find(type => MediaRecorder.isTypeSupported(type));
        const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
        this.recorder = recorder;
        const chunks = [];
        let size = 0;
        recorder.ondataavailable = event => {
          if (generation !== this.generation || !event.data.size) return;
          size += event.data.size;
          if (size > 10 * 1024 * 1024) return this.fail('Recording exceeded 10 MB. Try a shorter description.');
          chunks.push(event.data);
        };
        recorder.onerror = () => {
          if (generation === this.generation) this.fail('Recording failed. Check your microphone and try again.');
        };
        recorder.onstop = async () => {
          if (generation !== this.generation) return;
          this.release();
          this.recorder = null;
          const type = recorder.mimeType || chunks[0]?.type || mimeType || 'audio/webm';
          const blob = new Blob(chunks, { type });
          if (!blob.size) return this.fail('No audio captured. Speak into your microphone and try again.');
          const extension = type.includes('mp4') ? 'm4a' : type.includes('ogg') ? 'ogg' : 'webm';
          const form = new FormData();
          form.append('audio', blob, 'meal.' + extension);
          const controller = new AbortController();
          this.request = controller;
          this.update({ phase: 'processing' });
          try {
            const result = await this.api.dictateMeal(form, controller.signal);
            if (generation !== this.generation) return;
            this.update({ phase: 'review', result });
          } catch (error) {
            if (generation === this.generation && error.name !== 'AbortError') this.fail(error.message);
          } finally {
            if (this.request === controller) this.request = null;
          }
        };
        recorder.start(1000);
        const started = Date.now();
        this.update({ phase: 'recording', seconds: 0 });
        this.timer = setInterval(() => {
          const seconds = Math.floor((Date.now() - started) / 1000);
          this.update({ seconds: Math.min(seconds, 120) });
          if (seconds >= 120) this.stop();
        }, 1000);
      } catch (error) {
        if (generation !== this.generation) return;
        this.fail(error.name === 'NotAllowedError'
          ? 'Microphone access was denied. Allow it in your browser settings, then try again.'
          : error.name === 'NotFoundError' ? 'No microphone found. Connect one and try again.'
            : 'Could not start recording. Check your microphone and try again.');
      }
    }

    stop() {
      if (!this.recorder || this.recorder.state === 'inactive') return;
      clearInterval(this.timer);
      this.update({ phase: 'processing' });
      this.recorder.stop();
      this.release();
    }
  }
  window.MealDictation = MealDictation;
})();
