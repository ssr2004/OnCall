const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class FakeElement {
    constructor(tagName) {
        this.tagName = tagName;
        this.children = [];
        this.dataset = {};
        this.attributes = {};
        this.listeners = {};
        this.className = '';
        this.textContent = '';
        this.disabled = false;
        this.classList = {
            add: (...names) => {
                const values = new Set(this.className.split(/\s+/).filter(Boolean));
                names.forEach(name => values.add(name));
                this.className = [...values].join(' ');
            },
            remove: (...names) => {
                const blocked = new Set(names);
                this.className = this.className
                    .split(/\s+/)
                    .filter(name => name && !blocked.has(name))
                    .join(' ');
            }
        };
    }

    append(...children) {
        this.children.push(...children);
    }

    appendChild(child) {
        this.children.push(child);
        return child;
    }

    remove() {
        this.removed = true;
    }

    setAttribute(name, value) {
        this.attributes[name] = value;
    }

    addEventListener(name, listener) {
        this.listeners[name] = listener;
    }

    querySelector(selector) {
        if (!selector.startsWith('.')) return null;
        const className = selector.slice(1);
        for (const child of this.children) {
            if (child.className.split(/\s+/).includes(className)) return child;
            const nested = child.querySelector ? child.querySelector(selector) : null;
            if (nested) return nested;
        }
        return null;
    }
}

const appPath = path.resolve(__dirname, '..', 'static', 'app.js');
const source = fs.readFileSync(appPath, 'utf8');
const context = {
    console,
    setTimeout,
    clearTimeout,
    document: {
        createElement: tagName => new FakeElement(tagName),
        head: { appendChild: () => {} },
        addEventListener: () => {}
    }
};
context.globalThis = context;
vm.runInNewContext(
    `${source}\nglobalThis.__SuperBizAgentApp = SuperBizAgentApp;`,
    context,
    { filename: appPath }
);

async function run() {
    const App = context.__SuperBizAgentApp;
    const instance = Object.create(App.prototype);
    instance.apiBaseUrl = 'http://localhost:9900/api';

    let requestedUrl = '';
    let requestedOptions = null;
    context.fetch = async (url, options) => {
        requestedUrl = url;
        requestedOptions = options;
        return {
            ok: true,
            json: async () => ({
                incident_id: 'inc-grid/001',
                status: 'confirmed',
                persisted: true,
                message: '诊断已确认，并已写入长期情景记忆'
            })
        };
    };

    const report = new FakeElement('div');
    instance.renderIncidentActions(report, {
        incidentId: 'inc-grid/001',
        canConfirm: true,
        hasActiveAlerts: true
    });

    const actions = report.querySelector('.incident-memory-actions');
    const confirm = report.querySelector('.incident-memory-confirm');
    const reject = report.querySelector('.incident-memory-reject');
    const status = report.querySelector('.incident-memory-status');
    assert.ok(actions);
    assert.equal(confirm.textContent, '确认诊断');
    assert.equal(reject.textContent, '诊断不准确');

    await confirm.listeners.click();
    assert.equal(
        requestedUrl,
        'http://localhost:9900/api/aiops/incidents/inc-grid%2F001/confirm'
    );
    assert.equal(requestedOptions.method, 'POST');
    assert.equal(confirm.disabled, true);
    assert.equal(reject.disabled, true);
    assert.match(status.textContent, /写入长期情景记忆/);
    assert.ok(actions.className.includes('is-confirmed'));

    await assert.rejects(
        () => instance.submitIncidentDecision('inc-1', 'unknown'),
        /不支持的诊断反馈类型/
    );

    console.log('AIOPS_INCIDENT_FRONTEND_TEST_OK');
}

run().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
