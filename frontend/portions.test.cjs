const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const record = {
  fdcId: 101, description: 'Chicken', source: 'USDA',
  per_100g: { kcal: 165, protein_g: 31, carbs_g: 0, fat_g: 3.6 },
  portions: [{ label: '1 serving', grams: 140 }],
  default_portion: { label: '1 serving', grams: 140 },
};

function setup(api = {}) {
  const handlers = {}, queries = [];
  const root = { addEventListener: (event, fn) => { handlers[event] = fn; },
    querySelector: () => null, querySelectorAll: () => [] };
  const context = {
    window: { glyimpseAPI: { foodNutrients: async query => { queries.push(query); return record; }, ...api },
      addEventListener() {}, MealDictation: class {
        state = {};
        cancel() { this.state = {}; }
      } },
    document: { getElementById: () => root, addEventListener() {} },
  };
  // Expose the existing app functions inside this isolated test VM; skip network boot.
  const source = fs.readFileSync(__dirname + '/app.js', 'utf8')
    .replace(/  \/\/ -+ boot[\s\S]*$/, `
      window.testApp = { S, dictation, useDictatedMeal, dictationCard, foodDetail, mealRequest, setPortion };
    })();`);
  vm.runInNewContext(source, context);
  return { ...context.window.testApp, handlers, queries };
}

const flush = () => new Promise(resolve => setImmediate(resolve));

test('dictated weights and assumed servings reach the meal request with reminders only for assumptions', () => {
  const h = setup();
  const item = (quantity, grams, assumed) => ({
    name: 'Chicken', quantity, unit: quantity == null ? null : 'g', warnings: [],
    match: { fdcId: 101, description: 'Chicken' },
    nutrition: { ...record, serving: { grams, assumed, basis: assumed ? 'assumed standard serving' : 'stated weight',
      portion_index: assumed ? 0 : null, portion_count: assumed ? 1 : null } },
  });
  h.dictation.state.result = { transcript: '100g chicken', items: [item(100, 100, false)] };
  assert.doesNotMatch(h.dictationCard(), /double-check|Review portions|Portion needs review/);
  h.dictation.state.result.items.push(item(null, 140, true));
  assert.match(h.dictationCard(), /double-check this assumed amount/);
  h.useDictatedMeal();
  assert.equal(h.mealRequest().dietary_intake, 'Chicken 100 g, Chicken 140 g');
  assert.equal(h.S.builder.items[0].customGrams, 100);
  assert.doesNotMatch(h.foodDetail(h.S.builder.items[0], 0), /double-check|Compare this/);
  assert.match(h.foodDetail(h.S.builder.items[1], 1), /standard serving was assumed.*double-check/);
  h.setPortion(1, 'per100');
  assert.doesNotMatch(h.foodDetail(h.S.builder.items[1], 1), /double-check/);
});

test('typed weights are stripped from the lookup and preserved through USDA resolution', async () => {
  for (const [text, grams] of [['100g of chicken', 100], ['chicken 200 grams', 200],
    ['0.5 kg chicken', 500], ['.5 kg chicken', 500], ['1/2 kg chicken', 500]]) {
    const h = setup();
    h.handlers.change({ target: { dataset: { item: '0', part: 'name' }, value: text } });
    await flush();
    assert.equal(h.queries[0], 'chicken');
    assert.equal(h.mealRequest().foods[0].grams, grams);
    assert.doesNotMatch(h.foodDetail(h.S.builder.items[0], 0), /double-check/);
  }
});

test('a food typed without an amount uses a standard serving and prompts a double-check', async () => {
  const h = setup();
  h.handlers.change({ target: { dataset: { item: '0', part: 'name' }, value: 'chicken' } });
  await flush();
  assert.equal(h.mealRequest().foods[0].grams, 140);
  assert.match(h.foodDetail(h.S.builder.items[0], 0), /double-check the 140 g amount/);
  h.handlers.change({ target: { dataset: { item: '0', part: 'custom-grams' }, value: '75' } });
  assert.equal(h.mealRequest().foods[0].grams, 75);
  assert.doesNotMatch(h.foodDetail(h.S.builder.items[0], 0), /double-check/);
});

test('a failed nutrition lookup does not discard an explicitly typed weight', async () => {
  const h = setup({ foodNutrients: async () => { throw new Error('Unavailable'); } });
  h.handlers.change({ target: { dataset: { item: '0', part: 'name' }, value: '100g chicken' } });
  await flush();
  assert.equal(h.mealRequest().dietary_intake, 'chicken 100 g');
});
