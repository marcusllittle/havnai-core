// Offline DOM contract: no network or actual media. Run: node --test tests/dashboard_previews.test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function fixture(base = '/api') {
  const requests = [];
  class Element {
    constructor() { this.children = []; this.listeners = {}; }
    append(child) { this.children.push(child); child.parent = this; }
    replaceChildren(...children) { this.children = children; }
    remove() { this.parent.children = this.parent.children.filter(child => child !== this); }
    addEventListener(name, callback) { this.listeners[name] = callback; }
    set src(value) { requests.push(value); }
  }
  const context = vm.createContext({
    window: {}, document: { getElementById: () => new Element(), createElement: () => new Element() },
    console, setInterval: () => {},
  });
  const source = fs.readFileSync(path.join(__dirname, '../static/grid-feed.js'), 'utf8');
  // Suppress only dashboard startup. Exercise the production preview functions.
  vm.runInContext(source.replace(/^startDashboard\(\);$/m, ''), context);
  vm.runInContext(`preferredApiBase = ${JSON.stringify(base)}`, context);
  const card = new Element();
  const placeholder = new Element();
  card.append(placeholder);
  return { card, placeholder, requests, attach: value => context.attachApprovedPreview(card, value) };
}

test('approved reduced thumbnail resolves through API proxy and replaces placeholder only after load', () => {
  const f = fixture();
  f.attach('/v2/marketplace/listings/42/preview');
  assert.deepEqual(f.requests, ['/api/v2/marketplace/listings/42/preview']);
  const img = f.card.children[1];
  assert.equal(img.hidden, true);
  assert.equal(f.card.children[0], f.placeholder);
  img.listeners.load();
  assert.deepEqual(f.card.children, [img]);
  assert.equal(img.hidden, false);
  assert.equal(img.alt, 'Public listing preview');
});

test('denied, deleted or missing thumbnail leaves neutral placeholder without retrying original', () => {
  const f = fixture('');
  f.attach('/v2/marketplace/listings/7/preview');
  f.card.children[1].listeners.error();
  assert.deepEqual(f.card.children, [f.placeholder]);
  assert.deepEqual(f.requests, ['/v2/marketplace/listings/7/preview']);
});

test('missing, original, external and malformed URLs never request an image', () => {
  for (const value of [null, undefined, '', 42, '/static/outputs/job.png',
    'https://example.com/v2/marketplace/listings/1/preview', '//example.com/image',
    '/api/v2/marketplace/listings/1/preview', '/v2/marketplace/listings/0/preview',
    '/v2/marketplace/listings/1/preview?original=1', '/v2/marketplace/listings/../preview']) {
    const f = fixture();
    f.attach(value);
    assert.deepEqual(f.requests, [], String(value));
    assert.deepEqual(f.card.children, [f.placeholder]);
  }
});
