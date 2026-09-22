# 模块解读 01：小林面试笔记 RAG 问答系统

> 本文拆解本项目 `xiaolinrag/` 包，讲清「从 235 篇 markdown 笔记到一个可溯源的问答接口」这条链路。
> 全文基于本项目 `xiaolinrag/` 包（模块解读，供自研参考）。

## 1. 模块职责

系统分**离线构建链**与**在线查询链**，共享配置层与索引存储。

| 文件 | 职责 |
|------|------|
| `config.py` | 加载 `config.yaml`（非敏感）+ `.env`（密钥），校验后产出 `Config`；对外输出一律打码 |
| `corpus.py` | 发现并解析笔记页，与 `manifest.json` 核对数量，产出 `Page` |
| `chunking.py` | 父子切分：语义块 → 子块（检索）+ 父块（阅读） |
| `contextual.py` | Contextual Retrieval：为每个子块生成背景说明 |
| `embedder.py` | 向量化（SiliconFlow Qwen3-Embedding），批量 + 退避重试 + query 指令前缀 |
| `reranker.py` | 精排（SiliconFlow bge-reranker-v2-m3） |
| `llm.py` | 对话补全（OpenAI 兼容网关），错误统一翻译成中文提示 |
| `index_store.py` | 索引存取与两条检索通道：FAISS 向量 + JSONL 条目 + BM25 |
| `retrieval.py` | RRF 融合两路召回 |
| `prompts.py` | 提示词模板与 `[n]` 引用编号约定 |
| `pipeline.py` | 在线查询编排：召回 → 融合 → 精排 → 门控 → 生成 → 引用解析 |
| `eval.py` | 离线评估：Hit@K / MRR + LLM-as-a-Judge |
| `reuse.py` | 增量复用：把上一版索引当复用源，按内容寻址取回背景说明与向量（模块 05，见 `docs/05-多站点知识库.md`） |
| `cli.py` / `api.py` / `static/` | 命令行入口（build/status/serve/eval）与 Web 界面（模块 02 重做，见 `docs/02-前端与API.md`） |

**边界**：本包只读知识库，不抓取、不更新语料（那是 XiaoLinCrawler 的职责）；不做多模态检索、Query 改写、多轮对话。模块 05 加入的「增量」指的是**省掉对外部服务的重复调用**，索引产物仍是整体重建。

## 2. 核心数据结构

### `Page`（`corpus.py`）

一篇文章：`page_id`（`rag/4_chunking`，相对路径去扩展名）、`section`、`title`、`source_url`、`text`（已剥离 frontmatter 的正文）。

`page_id` 用相对路径而不是爬虫侧的 `id` 字段，是为了让索引里的每一条都能反查到磁盘文件。

### `ChildChunk` / `ParentChunk`（`chunking.py`）

```python
@dataclass
class ChildChunk:
    chunk_id: str      # "{page_id}#c{seq}"
    heading_path: str  # "4. RAG… > 📝 详细解析 > 粒度怎么定"
    child_text: str    # 以最近标题开头，供检索与展示
    parent_id: str     # 关联父块
    context_note: str  # Contextual Retrieval 背景，由 contextual.py 填充
    seq: int
```

`Page` → 一组 `ChildChunk` + 一组 `ParentChunk`，父子通过 ID 关联。子块是**检索单元**，父块是**阅读单元**。

### `IndexEntry` / `ScoredHit`（`index_store.py`）

`IndexEntry` 是落盘的一行，把子块、父块原文与来源元数据打平在一起——这样检索命中一条就能直接拿到引用所需的全部字段，不必回查磁盘。

```python
@property
def embed_text(self) -> str:
    """送入 Embedding 的文本：Contextual 背景 + 子块原文。"""
    if self.context_note:
        return f"{self.context_note}\n\n{self.child_text}"
    return self.child_text
```

子块原文原样保留（用于展示与喂给模型），只有**送去向量化的文本**才拼接背景说明——这是 Contextual Retrieval 的关键：改的是表示，不是内容。

### `AnswerResult` / `Citation`（`pipeline.py`）

`rejected` 标记本次是否被门控拒答；`citations` 是解析出的引用列表；`debug` 带双路召回、融合、重排分数，供验收与排障。

## 3. 关键流程

### 3.1 离线构建（`xiaolinrag build`）

```
load_pages / load_sites(kb_dir)           # 235 篇 / 2 个站点 / 10 个栏目，与索引清单核对
  ↓
ReuseSource.load(index_dir)               # 把上一版索引当复用源；读不到就是空源
  ↓
split_page(page, cfg.chunking) × 235      # → 7053 子块 / 2215 父块
  ↓
enrich(children, pages, cfg, reuse=reuse) # 内容没变的文章直接沿用背景说明，7053/7053
  ↓
向量化：命中取复用源的向量，未命中才调 embed_documents   # → (7053, 4096)，L2 归一化
  ↓
IndexStore.build(entries, vectors, meta).save(index_dir)   # 三件套整体重写
```

耗时实测：切分 1.6s；背景生成 1087s、向量化 1674s，**首次全量约 46 分钟**（视服务端速度浮动）。
之后再跑一次同一语料：背景命中 235/235 篇、向量命中 7053/7053，**4.4 秒**跑完；只改一篇文章的正文时 21 秒。
复用省掉的只是对外部服务的调用，索引产物仍是整体重建（详见 `docs/05-多站点知识库.md`）。

### 3.2 切分：从 markdown 到父子块（`chunking.py`）

```
markdown 解析 → 语义块序列（heading / para / code / list / table / quote）
  ↓ 按标题层级分组，维护标题栈得到 heading_path
  ↓ 组内按段落打包成子块：
      代码块 / 列表 / 表格 → 整块原子，永不截断
        └ 单块超过 child_max 时独占一个子块（允许超限）
        └ 普通段落超过 child_max 时按句子边界回退切分
      其余按 child_target(300) 打包，硬上限 child_max(500)
  ↓ 标题前缀计入长度预算，保证含前缀后仍不超限
  ↓ 子块按顺序窗口打包成父块（target 900 / max 1200），父子 ID 互指
```

实测：7053 个子块中 147 个超 500 字，**全部**是整块保留类型（代码 82、列表 62、表格 3），没有一个是段落被切坏。这一条是构造上保证的：超长段落会先按句子边界回退，回退出来的片段必不超限，只有原子块允许超限。父子包含性检查 0 失败。

### 3.3 在线查询（`pipeline.ask`）

```
embed_query(question)                      # Qwen3 检索指令前缀
  ├→ search_vector(vec, top_k=20, sections)   # FAISS IndexFlatIP
  └→ search_bm25(question, top_k=20, sections) # jieba + BM25Okapi
        ↓ fuse_rrf(vec_hits, kw_hits, k=60)    # Σ 1/(k+rank)，同块累加
        ↓ rerank(question, fused[:40])         # 精排，分数归一到 [0,1]
        ↓ 门控：max(rerank_score) < 0.3 → 拒答
        ↓ top5 子块 → 按 parent_id 去重 → 取父块原文
        ↓ llm.chat(SYSTEM_PROMPT, 带 [n] 编号的资料)
        ↓ parse_citations(answer, entries)     # [n] → Citation，越界记 dangling
```

### 3.4 三路行号对齐（`index_store.py`）

```
FAISS 行号  ==  entries.jsonl 行号  ==  BM25 文档序
```

由 `build` 一次性建立（`add_with_ids(vectors, arange(n))`），`load` 时校验 `index.ntotal == len(entries)`，不一致直接提示重建。三路任何一处漂移都会让「检索命中的向量」和「取出的原文」对不上，所以必须在加载期拦住。

### 3.5 门控与引用解析

门控放在**精排之后、生成之前**——这是唯一能拿到可靠相关度分数的位置。拒答时构造明确的文案（带上分数与阈值），`citations` 置空，绝不调用生成模型。

引用解析按 `[n]` 正则扫答案，映射回进入生成的资料列表（去重、保持首次出现顺序）；`n` 越界记入 `debug["dangling_citations"]`，供验收核查是否有悬空引用。

## 4. 要点小结

1. **为什么切分要引入父块，而不是直接把 chunk 调大？**
   调大 chunk 会让向量把多个话题压缩在一起，检索精度掉；调小又让模型读不到上下文。父子切割把「定位」和「阅读」拆成两个粒度，代价只是存储翻倍——在 2000 块这个量级完全不心疼。

2. **Contextual Retrieval 为什么按文章成批调用，而不是逐块调用？**
   逐块要 7053 次 LLM 调用，按文章成批只要 336 次（实测 1087 秒）。文章全文本来就要作为上下文传进去，一次调用里顺带为该篇所有子块生成背景，输入只传一遍。代价是需要解析结构化输出、并处理对齐失败——所以解析器有 JSON 与逐行两级回退，失败时该批降级为空背景，**不阻断构建**。这也是模块 05 的增量复用能按「文章」为单位命中的原因：输入只跟正文与子块序列有关。

3. **已经有 Prompt 约束「资料不足就说找不到」，为什么还要门控？**
   两者挡的不是同一类失败。门控挡的是「检索根本没召回有用的东西」——这时上下文是垃圾，模型再守规矩也只能在垃圾上作答；模块 01 用旧版评测集标定过阈值：库外问题精排最高分在 0.0015~0.044，库内问题最低 0.574，取 0.3 能把这一整段噪声干净地切掉（模块 05 换了评测集但没有重新标定）。Prompt 约束挡的是「召回内容相关但答不了这个问题」——比如问「明天北京天气」，知识库里有这句话（Function Calling 的示例），分数 0.85 过了门控，这时靠 Prompt 让模型说明「那是演示假数据」而不是编造天气。

4. **7053 条向量，用 numpy 直接矩阵乘比 FAISS 更快，为什么还用 FAISS？**
   这个量级 numpy 确实够快。选 FAISS 是为了保留向上的路径：换 IVF/HNSW 只改一行索引构造，检索代码不动；栏目过滤也能直接用 `IDSelectorBatch` 在索引层预过滤，而不是先检索再筛掉（后者会让某个栏目在小候选集里被其他栏目挤空）。三路行号对齐的设计也是为将来换索引留的接口。

5. **栏目过滤为什么用 `IDSelectorBatch` 预过滤，而不是检索后筛？**
   检索后筛会静默丢结果：如果 top-20 里 19 条都属于别的栏目，筛完只剩 1 条，而用户以为「这个栏目里只有 1 条相关内容」。预过滤让 FAISS 只在目标栏目的行里排序，返回数量与质量都可预期。

6. **子块为什么要在开头拼上最近的标题？**
   子块被单独拎出来时常常「没头没尾」——「粒度怎么定？」这一段的正文里可能一次都没出现「chunk」。拼上标题后，子块自身可读（引用展示时不用回原文找语境），BM25 也能匹配到标题里的术语。长度受预算约束（前缀最多占 `child_max` 的四分之一），超长标题会被截断，不会挤掉正文。

## 5. 测试绑定

**对应的测试文件：**

- `tests/test_config.py` —— 配置加载、环境变量优先级、各类非法配置的报错，以及密钥打码
- `tests/test_corpus.py` —— 语料发现、与 manifest 的数量/集合核对、frontmatter 解析与缺字段报错
- `tests/test_chunking.py` —— 标题路径、代码块与表格不被截断、子块⊆父块、字数上限、超长段落句级回退
- `tests/test_contextual.py` —— 背景说明的解析（JSON/逐行回退/乱码）、截断、失败降级、拆批
- `tests/test_index_store.py` —— 构建数量/维度校验、存取往返、行号不一致检测、栏目过滤、BM25 命中
- `tests/test_retrieval.py` —— RRF 计分、跨路去重、同分稳定排序
- `tests/test_pipeline.py` —— 门控通过/拒答两条路径、栏目透传、父块去重、引用解析与悬空检测

**怎么验证本模块：**

- 全部测试：`python -m pytest -q`（当前 181 passed；本模块相关的是 `test_chunking.py` / `test_pipeline.py` 等早期文件）
- 单文件：`python -m pytest tests/test_chunking.py -q`
- 单条用例：`python -m pytest tests/test_pipeline.py::test_gate_rejects_low_score -q`

**需要知道：**

- 全部单元测试都用 mock，不消耗 API 额度；**真实链路**（向量化质量、重排效果、生成与引用）由 `python -m xiaolinrag eval` 承担。
- `docs/testing.md` 是全部测试的入口与总表，可反查任意模块。
- 本模块**无存量失败**。
- 切分统计、门控阈值、检索指标这些数字来自真实语料的一次实测，语料或模型变化后需要重跑 `eval` 复核，不要当成恒定值。
