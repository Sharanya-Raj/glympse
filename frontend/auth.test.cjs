const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function setup(responses = []) {
  const calls = [], handlers = {}, saved = new Map();
  const root = { addEventListener: (event, fn) => { handlers[event] = fn; },
    querySelector: () => null, querySelectorAll: () => [] };
  const context = {
    window: { addEventListener() {}, MealDictation: class { state = {}; cancel() {} } },
    document: { getElementById: () => root, addEventListener() {} },
    localStorage: { getItem: k => saved.get(k), setItem: (k, v) => saved.set(k, v), removeItem: k => saved.delete(k) },
    location: { origin: 'http://test.invalid' }, clearTimeout, setTimeout,
    fetch: async (url, options) => {
      calls.push({ url, options, json: options.body ? JSON.parse(options.body) : null });
      const result = responses.shift();
      if (result instanceof Error) throw result;
      assert.ok(result, `Unexpected request: ${url}`);
      return { ok: result.status < 400, status: result.status, json: async () => result.data };
    }
  };
  vm.runInNewContext(fs.readFileSync(__dirname + '/api.js', 'utf8'), context);
  const source = fs.readFileSync(__dirname + '/app.js', 'utf8')
    .replace(/  \/\/ -+ boot[\s\S]*$/, `
      window.testApp = { S, signIn, recoverPassword, viewSignin };
    })();`);
  vm.runInNewContext(source, context);
  return { ...context.window.testApp, api: context.window.glyimpseAPI, calls, saved, root, handlers };
}

const form = (fields) => Object.fromEntries(Object.entries(fields).map(([k, value]) => [k, { value }]));
const ok = data => ({ status: 200, data });

test('sign-in offers email/password and recovery without an account confirmation field', () => {
  const h = setup();
  const html = h.viewSignin();
  assert.match(html, /name="email" type="email"/);
  assert.match(html, /name="password"/);
  assert.match(html, /Forgot password\?/);
  assert.doesNotMatch(html, /name="username"|name="confirmation"|name="code"|Confirmation code/);
});

test('email sign-in reaches the API and loads the account under its email', async () => {
  const h = setup([ok({ access_token: 'test-token' }), ok({ status: 'ok' }),
    ok({ email: 'qa+test@example.com', admin: false }), { status: 409, data: { detail: 'Upload glucose history first' } }]);
  await h.signIn(form({ email: ' qa+test@example.com ', password: 'secret' }));
  assert.deepEqual(h.calls[0].json, { email: 'qa+test@example.com', password: 'secret' });
  assert.equal(h.calls[0].url, '/sessions');
  assert.equal(h.calls[0].options.headers.Authorization, undefined);
  assert.equal(h.calls[2].options.headers.Authorization, 'Bearer test-token');
  assert.equal(h.S.user, 'qa+test@example.com');
  assert.equal(h.S.tab, 'today');
});

test('invalid email and duplicate submits do not call backend', async () => {
  const h = setup();
  await h.signIn(form({ email: 'username', password: 'secret' }));
  assert.match(h.S.signinError, /valid email/);
  h.S.signinBusy = true;
  await h.signIn(form({ email: 'qa@example.com', password: 'secret' }));
  assert.equal(h.calls.length, 0);
});

test('recovery sends code, resends, rejects mismatched passwords, then resets and returns to login', async () => {
  const h = setup([ok({ message: 'Code sent if eligible.' }), ok({ message: 'Code sent if eligible.' }),
    ok({ message: 'Password updated.' })]);
  h.S.authMode = 'forgot';
  await h.recoverPassword(form({ email: 'qa@example.com' }));
  assert.equal(h.S.authMode, 'reset');
  assert.equal(h.calls[0].url, '/password/forgot');
  assert.match(h.viewSignin(), /name="code"/);
  const reset = form({ email: 'qa@example.com', code: ' 123456 ', password: 'new-password', repeatPassword: 'different' });
  await h.recoverPassword(reset);
  assert.match(h.S.signinError, /do not match/);
  assert.equal(h.calls.length, 1);
  await h.recoverPassword(reset, true);
  assert.equal(h.calls[1].url, '/password/forgot');
  reset.repeatPassword.value = 'new-password';
  h.api.setToken('old-token');
  await h.recoverPassword(reset);
  assert.equal(h.calls[2].url, '/password/reset');
  assert.deepEqual(h.calls[2].json, { email: 'qa@example.com', code: '123456', password: 'new-password' });
  assert.equal(h.calls[2].options.headers.Authorization, undefined);
  assert.equal(h.api.token, null);
  assert.equal(h.S.authMode, 'signin');
  assert.match(h.S.authNotice, /Password updated/);
  assert.equal(h.S.signinBusy, false);
});

test('expired code and network errors stay recoverable and do not sign out another session', async () => {
  const h = setup([{ status: 422, data: { detail: 'That code has expired.' } }, new Error('network'),
    { status: 401, data: { detail: 'Unable to reset password.' } }]);
  h.S.authMode = 'reset';
  h.api.setToken('existing-token');
  const reset = form({ email: 'qa@example.com', code: '123456', password: 'new-password', repeatPassword: 'new-password' });
  await h.recoverPassword(reset);
  assert.match(h.S.signinError, /expired/);
  assert.equal(h.S.authMode, 'reset');
  await h.recoverPassword(reset, true);
  assert.match(h.S.signinError, /reach the Glyimpse server/);
  assert.equal(h.S.signinBusy, false);
  await h.recoverPassword(reset);
  assert.equal(h.api.token, 'existing-token');
  assert.equal(h.S.authMode, 'reset');
});
