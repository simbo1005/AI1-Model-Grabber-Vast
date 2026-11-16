const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const test = require('node:test');
const source = fs.readFileSync(require('node:path').join(__dirname, '../launcher/static/app.js'), 'utf8');
const fn = source.match(/function estimatedTransferMinutes\([^]*?^}/m)[0];
const context = vm.createContext({});vm.runInContext(fn, context);
test('transfer estimate handles empty selections, invalid input and minute rounding', () => {
  const estimate=context.estimatedTransferMinutes;
  assert.equal(estimate(0),0);assert.equal(estimate(-1),0);assert.equal(estimate(Infinity),0);
  assert.equal(estimate(1),1);assert.equal(estimate(57e6*60),1);
  assert.equal(estimate(57e6*60+1),2);assert.equal(estimate(120,1),2);
  assert.equal(estimate(57e6*60,0),1);
});
