const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { File } = require('node:buffer');
function setup(responses) {
  const calls = [];
  const context = { window: {}, localStorage: { getItem: () => 'synthetic-token', removeItem() {}, setItem() {} }, location: { origin: 'http://local.test' }, FormData, AbortSignal,
    fetch: async (url, options) => { calls.push({url, options}); const result = responses.shift(); if (result instanceof Error) throw result; return { ok: result.status < 400, status: result.status, json: async () => result.data }; } };
  vm.runInNewContext(fs.readFileSync('frontend/api.js','utf8'), context);
  return { api: context.window.glyimpseAPI, calls };
}
const intent = {status:201,data:{image_id:'abc',upload:{url:'https://synthetic.s3.test',fields:{key:'food-images/synthetic/pending/abc',policy:'signed'}}}};
test('uploads direct to S3 without Cognito token and validates before returning', async () => {
  const {api,calls}=setup([intent,{status:204},{status:200,data:{image_id:'abc',status:'validated'}}]);
  assert.equal((await api.uploadFoodImage(new File(['x'],'meal.png',{type:'image/png'}))).status,'validated');
  assert.equal(calls[1].options.headers,undefined);
  assert.equal(calls[1].options.credentials,'omit');
  assert.equal([...calls[1].options.body.keys()].at(-1),'file');
  assert.equal(calls[2].url,'/food/images/abc/complete');
});
test('rejects unsupported files before network', async () => {
  const {api,calls}=setup([]);
  await assert.rejects(api.uploadFoodImage(new File(['x'],'bad.svg',{type:'image/svg+xml'})));
  assert.equal(calls.length,0);
});
test('renews a rejected presigned policy once', async () => {
  const {api,calls}=setup([intent,{status:403},intent,{status:204},{status:200,data:{image_id:'abc'}}]);
  await api.uploadFoodImage(new File(['x'],'meal.png',{type:'image/png'}));
  assert.equal(calls.length,5);
});
test('failed upload never completes or attaches', async () => {
  const {api,calls}=setup([intent,new Error('network')]);
  await assert.rejects(api.uploadFoodImage(new File(['x'],'meal.png',{type:'image/png'})),/upload failed/);
  assert.equal(calls.length,2);
});
test('retrieval requests authenticated fresh URL', async () => {
  const {api,calls}=setup([{status:200,data:{url:'https://synthetic.s3.test/read',expires_in:120}}]);
  await api.foodImageUrl('abc');
  assert.equal(calls[0].options.headers.Authorization,'Bearer synthetic-token');
});
