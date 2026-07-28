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
    document: {
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

async function run() {
    const App = context.__SuperBizAgentApp;
    const instance = Object.create(App.prototype);
    instance.apiBaseUrl = 'http://localhost:9900/api';
    instance.sessionId = 'frontend-aiops-context-test';
    instance.currentChatHistory = [];
    instance.currentAIOpsContext = '';

    const report = [
        '# 告警分析报告',
        '## 整体评估',
        'GridDataSyncServiceDown 导致服务健康检查失败。',
        '## 关键发现',
        '服务指标端点无法抓取。',
        '## 风险评估',
        '当前风险等级为严重。'
    ].join('\n\n');

    instance.currentChatHistory.push({
        type: 'assistant',
        content: report,
        source: 'aiops'
    });
    assert.equal(instance.restoreAIOpsContextFromHistory(), report);

    let requestedUrl = '';
    let requestedBody = null;
    context.fetch = async (url, options) => {
        requestedUrl = url;
        requestedBody = JSON.parse(options.body);
        return {
            ok: true,
            json: async () => ({
                code: 200,
                message: 'success',
                data: {
                    success: true,
                    answer: '报告说明电网数据同步服务不可用。'
                }
            })
        };
    };

    const visibleMessages = [];
    instance.addLoadingMessage = () => null;
    instance.addMessage = (type, content) => {
        visibleMessages.push({ type, content });
        return null;
    };

    const userQuestion = '上面的报告说明了什么？';
    await instance.sendQuickMessage(userQuestion);

    assert.equal(requestedUrl, 'http://localhost:9900/api/chat');
    assert.equal(requestedBody.Id, 'frontend-aiops-context-test');
    assert.match(requestedBody.Question, /<AIOPS_DIAGNOSIS_REPORT>/);
    assert.match(requestedBody.Question, /GridDataSyncServiceDown/);
    assert.match(requestedBody.Question, /上面的报告说明了什么/);
    assert.match(requestedBody.Question, /不要要求用户再次提供报告/);
    assert.notEqual(requestedBody.Question, userQuestion);
    assert.deepEqual(visibleMessages, [
        {
            type: 'assistant',
            content: '报告说明电网数据同步服务不可用。'
        }
    ]);

    const inferred = instance.findLatestAIOpsContext([
        {
            type: 'assistant',
            content: report
        }
    ]);
    assert.equal(inferred, report, '旧历史记录即使没有 source 字段也应恢复报告');

    instance.currentAIOpsContext = '';
    instance.currentChatHistory = [];
    assert.equal(
        instance.buildQuestionWithAIOpsContext('普通问题'),
        '普通问题',
        '没有 AIOps 报告时不应改写普通问题'
    );

    console.log('AIOPS_CONTEXT_FRONTEND_TEST_OK');
}

run().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
