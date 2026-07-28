const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const appPath = path.resolve(__dirname, '..', 'static', 'app.js');
const source = fs.readFileSync(appPath, 'utf8');

const context = {
    console,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    document: {
        hidden: false,
        createElement: () => ({}),
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

function createStatusLink() {
    const classes = new Set(['alert-status-link', 'alert-status-loading']);
    const attributes = {};
    return {
        classList: {
            add: (...names) => names.forEach(name => classes.add(name)),
            remove: (...names) => names.forEach(name => classes.delete(name)),
            contains: name => classes.has(name)
        },
        dataset: {},
        title: '',
        setAttribute: (name, value) => {
            attributes[name] = value;
        },
        getAttribute: name => attributes[name]
    };
}

async function run() {
    const App = context.__SuperBizAgentApp;
    const instance = Object.create(App.prototype);
    instance.apiBaseUrl = 'http://localhost:9900/api';
    instance.alertStatusRefreshInFlight = false;
    instance.alertStatusLink = createStatusLink();
    instance.alertStatusLabel = { textContent: '' };
    instance.alertStatusCount = { textContent: '', hidden: true };

    instance.updateAlertStatus({ success: true, status: 'healthy', total: 0, alerts: [] });
    assert.equal(instance.alertStatusLabel.textContent, '运行正常');
    assert.equal(instance.alertStatusCount.hidden, true);
    assert.ok(instance.alertStatusLink.classList.contains('alert-status-healthy'));

    instance.updateAlertStatus({
        success: true,
        status: 'pending',
        total: 2,
        pending: 2,
        alerts: [{ alert_name: 'GridTelemetryQueueBacklog' }]
    });
    assert.equal(instance.alertStatusLabel.textContent, '告警确认中');
    assert.equal(instance.alertStatusCount.textContent, '2');
    assert.equal(instance.alertStatusCount.hidden, false);
    assert.ok(instance.alertStatusLink.classList.contains('alert-status-pending'));

    instance.updateAlertStatus({
        success: true,
        status: 'firing',
        total: 3,
        firing: 1,
        pending: 2,
        alerts: [{ alert_name: 'GridCommunicationInterrupted' }]
    });
    assert.equal(instance.alertStatusLabel.textContent, '活动告警');
    assert.equal(instance.alertStatusCount.textContent, '3');
    assert.ok(instance.alertStatusLink.classList.contains('alert-status-firing'));
    assert.match(instance.alertStatusLink.title, /GridCommunicationInterrupted/);
    assert.match(instance.alertStatusLink.getAttribute('aria-label'), /Prometheus/);

    let requestedUrl = '';
    let requestedOptions = null;
    context.fetch = async (url, options) => {
        requestedUrl = url;
        requestedOptions = options;
        return {
            ok: true,
            json: async () => ({ success: true, status: 'healthy', total: 0, alerts: [] })
        };
    };
    await instance.refreshAlertStatus();
    assert.equal(requestedUrl, 'http://localhost:9900/api/aiops/alert-status');
    assert.equal(requestedOptions.cache, 'no-store');
    assert.equal(instance.alertStatusLabel.textContent, '运行正常');

    instance.updateAlertStatus({ success: false, message: 'monitor unavailable' });
    assert.equal(instance.alertStatusLabel.textContent, '监控连接异常');
    assert.ok(instance.alertStatusLink.classList.contains('alert-status-unavailable'));

    console.log('AIOPS_ALERT_STATUS_FRONTEND_TEST_OK');
}

run().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
