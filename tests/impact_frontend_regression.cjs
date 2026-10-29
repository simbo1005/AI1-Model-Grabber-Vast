const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const extensions = [];
const origin = {
  id: "origin",
  outputs: [{type: "IMAGE"}],
  connect(slot, target, input) {
    assert.equal(target, this, "connect must get the node object, not a string ID");
    assert.equal(input, "input1");
    return null; // LiteGraph's existing self-connection guard
  }
};
const app = {
  configuringGraph: false,
  graph: {getNodeById() {return origin;}, _nodes_by_id: {}},
  registerExtension(ext) {extensions.push(ext);}
};
const raw = fs.readFileSync(process.argv[2], "utf8");
const code = raw.replace(/^import .*;\r?\n/gm, "")
  .replace(/^export \{[^\n]*\};?\r?\n/gm, "")
  .replace(/^export /gm, "");
const context = {
  app, ComfyApp: {}, ComfyDialog: class {}, $el() {},
  api: {addEventListener() {}, async fetchApi() {return {async json() {return {data: []};}};}},
  isBeforeFrontendVersion() {return false;}, customAlert() {},
  window: {}, console, Error, URLSearchParams, Image: class {}
};
vm.runInNewContext(code, context, {filename: "impact-pack.js"});
assert.equal(extensions.length, 1);
const extension = extensions[0];
function throughManyFrames(fn, depth) {
  return depth ? throughManyFrames(fn, depth - 1) : fn();
}
function loadGraphData(fn) {return throughManyFrames(fn, 35);}
function makeNodeType() {
  return class Node {
    constructor() {
      this.inputs = [
        {name: "input1", type: "*", link: 1},
        {name: "select", type: "INT"},
        {name: "sel_mode", type: "BOOLEAN"},
        {name: "input2", type: "*", link: 2},
        {name: "input3", type: "*", link: null}
      ];
      this.outputs = [{name: "output", type: "*", links: []}];
      this.widgets = [{value: 1, options: {max: 2}}];
    }
    addInput(name, type) {this.inputs.push({name, type, link: null});}
    removeInput(index) {this.inputs.splice(index, 1);}
  };
}
(async () => {
  const initialStackLimit = Error.stackTraceLimit;
  // Exercise every dynamic input class sharing the problematic code path.
  for (const name of [
    "ImpactSwitch", "LatentSwitch", "SEGSSwitch",
    "ImpactMakeImageList", "ImpactMakeImageBatch", "ImpactMakeMaskList",
    "ImpactMakeMaskBatch", "ImpactMakeAnyList", "CombineRegionalPrompts",
    "ImpactCombineConditionings", "ImpactConcatConditionings", "ImpactSEGSConcat"
  ]) {
    const Type = makeNodeType();
    await extension.beforeRegisterNodeDef(Type, {name}, app);
    const node = new Type();
    const initialInputs = JSON.stringify(node.inputs);
    const link = {origin_id: "origin", origin_slot: 0, target_slot: 0, type: "IMAGE"};
    app.configuringGraph = true;
    for (let reload = 0; reload < 3; reload++) {
      node.onConnectionsChange(1, 0, true, link);
      node.onConnectionsChange(1, 3, true, {...link, target_slot: 3});
      assert.equal(JSON.stringify(node.inputs), initialInputs, name + " changed restored slots");
    }
    app.configuringGraph = false;
    // Older frontend fallback: loadGraphData is >10 frames away.
    loadGraphData(() => node.onConnectionsChange(1, 0, true, link));
    assert.equal(JSON.stringify(node.inputs), initialInputs, name + " missed a deep restore stack");
    assert.equal(Error.stackTraceLimit, initialStackLimit, "global stack limit was not restored");
    // Real user connections must still add the normal spare input.
    node.onConnectionsChange(1, 0, true, link);
    assert.equal(node.inputs.length, 6, name + " no longer handles user connections");
  }
  const Inverse = makeNodeType();
  await extension.beforeRegisterNodeDef(Inverse, {name: "ImpactInversedSwitch"}, app);
  const inverse = new Inverse();
  inverse.outputs = [
    {name: "output1", type: "*", links: [], slot_index: 0},
    {name: "output2", type: "*", links: [], slot_index: 1}
  ];
  inverse.removeOutput = (index) => {inverse.outputs.splice(index, 1);};
  app.configuringGraph = true;
  inverse.onConnectionsChange(2, 1, false, {origin_slot: 1});
  assert.equal(inverse.outputs.length, 2, "restore removed an inverse-switch output");
  app.configuringGraph = false;
  inverse.onConnectionsChange(2, 1, false, {origin_slot: 1});
  assert.equal(inverse.outputs.length, 1, "manual disconnect failed to remove the unused output");
  assert.equal(Error.stackTraceLimit, initialStackLimit);
  console.log("PASS: pinned Impact Pack JS; 12 dynamic node classes, repeated restore, deep stack fallback, manual connections and inverse-switch outputs.");
})().catch(error => {console.error(error); process.exitCode = 1;});
