const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { TextDecoder, TextEncoder } = require('node:util');

const appPath = path.resolve(__dirname, '..', 'static', 'app.js');
const source = fs.readFileSync(appPath, 'utf8');

const context = {
    console,
    TextDecoder,
    TextEncoder,
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

const events = [
    {
        type: 'status',
        stage: 'fetching_alerts',
        message: '正在从 Prometheus 获取电网业务活动告警...'
    },
    {
        type: 'plan',
        stage: 'plan_created',
        message: '执行计划已制定，共 2 个步骤',
        plan: ['查询 Prometheus 活动告警', '生成诊断报告'],
        completed_steps: 0,
        total_steps: 2
    },
    {
        type: 'step_complete',
        stage: 'step_executed',
        message: '步骤执行完成 (1/2)',
        current_step: '查询 Prometheus 活动告警',
        completed_steps: 1,
        total_steps: 2
    },
    {
        type: 'status',
        stage: 'replanner',
        message: '评估完成，正在执行下一步骤 (1/2)...'
    },
    {
        type: 'report',
        stage: 'final_report',
        message: '诊断流程完成 (2/2)',
        report: '# 告警分析报告\n\n当前没有活动告警。',
        completed_steps: 2,
        total_steps: 2,
        incident_id: 'inc-frontend-test',
        can_confirm: true,
        has_active_alerts: true,
        incident_status: 'pending'
    },
    {
        type: 'complete',
        stage: 'diagnosis_complete',
        message: '诊断流程完成 (2/2)',
        completed_steps: 2,
        total_steps: 2
    }
];

const payload = events
    .map(event => `event: message\ndata: ${JSON.stringify(event)}\n\n`)
    .join('');
const encodedPayload = new TextEncoder().encode(payload);
let delivered = false;

context.fetch = async () => ({
    ok: true,
    body: {
        getReader: () => ({
            read: async () => {
                if (delivered) return { done: true, value: undefined };
                delivered = true;
                return { done: false, value: encodedPayload };
            },
            releaseLock: () => {}
        })
    }
});

async function run() {
    const App = context.__SuperBizAgentApp;
    const instance = Object.create(App.prototype);
    instance.apiBaseUrl = 'http://localhost:9900/api';
    instance.sessionId = 'frontend-progress-test';

    const progressSnapshots = [];
    let finalContent = '';
    let finalDetails = [];
    let finalIncidentMeta = null;
    instance.updateAIOpsProgressCard = (_element, state) => {
        progressSnapshots.push({
            transientStatus: state.transientStatus,
            completedCount: state.completedCount,
            totalSteps: state.totalSteps,
            plan: [...state.plan]
        });
    };
    instance.updateAIOpsMessage = (_element, content, details, incidentMeta) => {
        finalContent = content;
        finalDetails = details;
        finalIncidentMeta = incidentMeta;
    };

    await instance.sendAIOpsRequest({});

    assert.match(finalContent, /诊断流程完成 \(2\/2\)/);
    assert.match(finalContent, /# 告警分析报告/);
    assert.doesNotMatch(finalContent, /执行计划/);
    assert.doesNotMatch(finalContent, /步骤执行完成/);
    assert.doesNotMatch(finalContent, /⏳/);
    assert.doesNotMatch(finalContent, /正在从 Prometheus 获取/);
    assert.doesNotMatch(finalContent, /正在执行下一步骤/);
    assert.deepEqual(
        [...finalDetails],
        ['查询 Prometheus 活动告警', '生成诊断报告']
    );
    assert.equal(finalIncidentMeta.incidentId, 'inc-frontend-test');
    assert.equal(finalIncidentMeta.canConfirm, true);
    assert.equal(finalIncidentMeta.hasActiveAlerts, true);
    assert.ok(
        progressSnapshots.some(snapshot => snapshot.transientStatus.includes('正在从 Prometheus 获取')),
        '获取告警状态应在执行期间临时显示'
    );
    assert.ok(
        progressSnapshots.some(snapshot => snapshot.transientStatus.includes('正在执行下一步骤')),
        'Replanner 状态应在执行期间临时显示'
    );
    assert.ok(
        progressSnapshots.some(snapshot => snapshot.completedCount === 1 && snapshot.totalSteps === 2),
        '进度卡应显示单调递增的完成数量'
    );
    assert.equal(
        instance.summarizeAIOpsStep(
            '步骤2: 使用query_grid_service_status工具查询最近一小时的服务健康状态。参数：无需额外参数。'
        ),
        '查询最近一小时的服务健康状态。'
    );

    console.log('AIOPS_FRONTEND_PROGRESS_TEST_OK');
}

run().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
