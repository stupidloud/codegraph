#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');

const serverRoot = path.resolve(__dirname, '..');
const targetProjectRoot = process.argv[2]
  ? path.resolve(process.argv[2])
  : process.cwd();

const serverArgs = [path.join(serverRoot, 'dist/bin/codegraph.js'), 'serve', '--mcp'];
const child = spawn(process.execPath, serverArgs, {
  cwd: targetProjectRoot,
  stdio: ['pipe', 'pipe', 'pipe'],
});

let responseCount = 0;
const pendingIds = new Set([1, 100, 101, 102, 103]);
let shutdownTimer;

function finishIfDone() {
  if (pendingIds.size > 0 || child.killed) return;
  clearTimeout(shutdownTimer);
  process.stderr.write('[done] all responses received; terminating MCP server\n');
  child.kill('SIGTERM');
}

function send(id, method, params) {
  pendingIds.add(id);
  child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
}

child.stdout.on('data', (chunk) => {
  const text = chunk.toString();
  process.stdout.write(`[stdout] ${text}`);

  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const message = JSON.parse(trimmed);
      if (Object.prototype.hasOwnProperty.call(message, 'id')) {
        responseCount++;
        pendingIds.delete(message.id);
        finishIfDone();
      }
    } catch {
      // Non-JSON stdout is still printed above for diagnostics.
    }
  }
});

child.stderr.on('data', (chunk) => {
  process.stderr.write(`[stderr] ${chunk.toString()}`);
});

child.on('error', (error) => {
  process.stderr.write(`[child error] ${error.stack || error.message}\n`);
});

child.on('exit', (code, signal) => {
  process.stderr.write(
    `[exit] code=${code} signal=${signal} responses=${responseCount} pending=${Array.from(pendingIds).join(',')}\n`
  );
});

send(1, 'initialize', {
  protocolVersion: '2024-11-05',
  capabilities: {},
  clientInfo: { name: 'debug-mcp-concurrency', version: '0' },
  rootUri: `file://${targetProjectRoot}`,
});

setTimeout(() => {
  const tasks = [
    '管理员登录、后台鉴权和权限校验在哪里实现？',
    '文件上传、附件保存、上传异常处理相关代码在哪里？',
    '短信验证码发送和校验、手机验证相关代码在哪里？',
    '生成 CRUD 控制器、模型、视图的命令行代码在哪里？',
  ];

  for (let i = 0; i < tasks.length; i++) {
    send(100 + i, 'tools/call', {
      name: 'codegraph_context',
      arguments: {
        task: tasks[i],
        maxNodes: 10,
        includeCode: false,
      },
    });
  }
}, 500);

shutdownTimer = setTimeout(() => {
  if (!child.killed) {
    process.stderr.write('[timeout] terminating MCP server after 60s\n');
    child.kill('SIGTERM');
  }
}, 60_000);
