# CodeGraph

> 为 Claude Code 提供代码知识图谱、符号搜索、调用关系分析和可选的 Gemini 语义搜索。

[English](./README.md) · [简体中文](./README.zh-CN.md)

## 这是什么

CodeGraph 会在项目本地生成 `.codegraph/codegraph.db`，把代码解析成可查询的知识图谱：

- 符号：函数、方法、类、接口、模块、路由等。
- 关系：调用、导入、继承、实现、引用等。
- 查询：全文搜索、调用链、影响面、上下文构建。
- 同步：MCP server 可监听文件变化并增量更新索引。
- 语义搜索：可选启用 Gemini embedding；向量和索引仍保存在本地 SQLite。

默认结构索引是本地的。只有启用 `semanticSearch` 时，生成 embedding 和语义查询 embedding 会调用 Gemini API。

## 安装

```bash
npm install -g @stupidloud/codegraph
```

也可以直接运行交互安装器：

```bash
npx @stupidloud/codegraph
```

交互安装器会配置 Claude Code 的 MCP server，并可选择初始化当前项目。

## 初始化项目

```bash
cd your-project
codegraph init -i
```

初始化时会询问是否启用 Gemini 语义搜索：

```text
Enable Gemini semantic search?
Gemini API key
```

如果选择启用，会写入 `.codegraph/config.json`：

```json
{
  "semanticSearch": {
    "enabled": true,
    "provider": "gemini",
    "apiKey": "YOUR_GEMINI_API_KEY",
    "model": "gemini-embedding-2",
    "outputDimensionality": 768,
    "batchSize": 32
  }
}
```

如果项目已经初始化过，也可以手动编辑 `.codegraph/config.json` 后重新索引：

```bash
codegraph index -f
```

## 常用命令

```bash
codegraph init [path]             # 初始化项目
codegraph index [path]            # 全量索引
codegraph index -f [path]         # 清空并重新索引
codegraph sync [path]             # 增量同步
codegraph status [path]           # 查看索引状态
codegraph query <search>          # 搜索符号
codegraph context <task>          # 为任务构建相关代码上下文
codegraph affected [files...]     # 查找受变更影响的测试文件
codegraph serve --mcp             # 启动 MCP server
codegraph visualize [path]        # 打开可视化界面
```

示例：

```bash
codegraph query AuthService
codegraph context "where is auth token refresh handled"
codegraph affected src/auth.ts
```

## Gemini 语义搜索如何工作

启用语义搜索后：

1. `codegraph index` 会为函数、方法、类、接口、模块、组件等节点生成 embedding。
2. embedding 通过 Gemini `batchEmbedContents` API 批量生成。
3. 向量保存在本地 SQLite 的 `vectors` 表。
4. 查询时，`codegraph context` 会为查询文本生成 query embedding。
5. 搜索优先使用 `sqlite-vss`，不可用时回退到本地 brute-force cosine。
6. 本地记录 `content_hash`，未变化的节点会跳过重新生成 embedding。

这个设计避免了本地模型下载和 ONNX/WASM 推理开销，但会把用于 embedding 的节点文本发送给 Gemini。不要在不可信项目或不允许外发源码的环境中启用语义搜索。

## Claude Code MCP 使用

启动 MCP server：

```bash
codegraph serve --mcp
```

Claude Code 可用的主要工具：

| 工具 | 用途 |
|---|---|
| `codegraph_search` | 按名称搜索符号 |
| `codegraph_context` | 为任务构建相关代码上下文 |
| `codegraph_callers` | 查找谁调用了某个符号 |
| `codegraph_callees` | 查找某个符号调用了什么 |
| `codegraph_impact` | 分析修改影响面 |
| `codegraph_node` | 查看单个符号详情和源码 |
| `codegraph_files` | 查看索引后的文件结构 |
| `codegraph_status` | 查看索引健康状态 |

## 作为库使用

```ts
import CodeGraph from '@stupidloud/codegraph';

const cg = await CodeGraph.init('/path/to/project', {
  config: {
    semanticSearch: {
      enabled: true,
      provider: 'gemini',
      apiKey: process.env.GEMINI_API_KEY,
      model: 'gemini-embedding-2',
      outputDimensionality: 768,
      batchSize: 32,
    },
  },
});

await cg.indexAll();

const results = cg.searchNodes('UserService');
const context = await cg.buildContext('fix login bug', {
  maxNodes: 20,
  includeCode: true,
  format: 'markdown',
});

cg.close();
```

## 配置

`.codegraph/config.json` 控制索引行为：

```json
{
  "version": 1,
  "languages": ["typescript", "javascript"],
  "exclude": ["node_modules/**", "dist/**", "build/**", "*.min.js"],
  "frameworks": [],
  "maxFileSize": 1048576,
  "extractDocstrings": true,
  "trackCallSites": true,
  "semanticSearch": {
    "enabled": false,
    "provider": "gemini",
    "model": "gemini-embedding-2",
    "outputDimensionality": 768,
    "batchSize": 32
  }
}
```

| 配置项 | 说明 |
|---|---|
| `languages` | 要索引的语言；为空时自动检测 |
| `exclude` | 排除的 glob 模式 |
| `frameworks` | 框架提示，用于更好的路由/引用解析 |
| `maxFileSize` | 跳过超过该大小的文件 |
| `extractDocstrings` | 是否提取文档注释 |
| `trackCallSites` | 是否记录调用位置 |
| `semanticSearch.enabled` | 是否启用 Gemini 语义搜索 |
| `semanticSearch.apiKey` | Gemini API key |
| `semanticSearch.model` | 默认 `gemini-embedding-2` |
| `semanticSearch.outputDimensionality` | 默认 `768` |
| `semanticSearch.batchSize` | 每批生成多少个节点 embedding |

## 支持语言

TypeScript、JavaScript、Python、Go、Rust、Java、C#、PHP、Ruby、C、C++、Swift、Kotlin、Dart、Svelte、Vue、Liquid、Pascal/Delphi、Scala 等。

## 注意事项

- Node.js 需要 `>=18 <25`。
- 第一次启用语义搜索后建议运行 `codegraph index -f`。
- `sqlite-vss` 是可选依赖；不可用时会自动回退到 brute-force cosine。
- npm 包名是 `@stupidloud/codegraph`。
