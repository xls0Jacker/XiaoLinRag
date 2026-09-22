# 测试说明

> 本文件是全部测试的入口与总表：怎么跑、每个测试文件守护什么、任意模块怎么反查测试。
> 模块实现细节见 `docs/01-rag问答系统.md`、`docs/02-前端与API.md`、`docs/03-渲染与图片.md`、
> `docs/05-多站点知识库.md` 与 `docs/06-多Query扩展.md`。

## 怎么跑

```bash
python -m pytest -q                # 全部（当前 213 passed）
python -m pytest tests/test_chunking.py -q            # 单个文件
python -m pytest tests/test_pipeline.py::test_gate_rejects_low_score -q   # 单条用例
```

全部单元测试**使用 mock，不发起任何网络请求**，不消耗 API 额度，可在离线环境运行。
例外是 `tests/test_frontend_js.py`：它不起网络，但需要 `node` 才能执行 `static/app.js`；未安装 node 时整份自动跳过。

测试要在项目的 conda 环境里跑（`faiss` 等依赖不在 base 环境）：

```bash
conda activate xiaolinrag && python -m pytest -q
```

真实链路（向量化质量、重排效果、生成忠实度）不由单元测试覆盖，由离线评估承担：

```bash
python -m xiaolinrag eval              # 检索层 Hit@K/MRR + 生成层 LLM 裁判（消耗 API）
python -m xiaolinrag eval --no-judge   # 只跑检索层
```

## 测试文件 → 功能点映射

| 测试文件 | 用例数 | 守护的功能点 |
|----------|--------|--------------|
| `tests/test_config.py` | 14 | 配置 YAML 加载、`.env` 与环境变量优先级、必填项与数值约束校验、网关额外请求头、密钥打码不泄漏、**multi_query 小节加载/缺省/校验/打码** |
| `tests/test_corpus.py` | 12 | 新索引格式（按站点分组的栏目清单）解析、递归发现章节子目录、assets/ 下的 markdown 不算语料、与索引的集合核对（缺文件/多文件/缺栏目）、frontmatter 解析与缺字段报错、站点/栏目展示名与声明顺序 |
| `tests/test_chunking.py` | 11 | 标题路径生成、代码块/表格/列表整块保留、子块⊆父块、父子 ID 双向关联、子块与父块字数上限、超长段落句级回退、仅标题页不产出 |
| `tests/test_contextual.py` | 17 | 背景说明解析（JSON / 夹杂散文 / 逐行回退 / 乱码 / 空输入）、生成后截断、模型失败与输出不可解析时降级为空背景、关闭开关时整体跳过、超 30 个子块拆批、**命中复用源时不调用模型**、复用条数不符时重算 |
| `tests/test_reuse.py` | 11 | 增量复用：同样内容命中、正文变化未命中、切分参数变化换键、对话/向量模型变化整批关闭、标题变化不影响命中、条目顺序无关、整篇空说明视为未命中、索引缺失或损坏时降级为空源、向量数不符只关掉向量复用 |
| `tests/test_index_store.py` | 16 | 构建时的数量与维度校验、存取往返、缺文件与行数漂移检测、栏目过滤（向量与 BM25 两路）、BM25 关键词命中、空查询、向量维度不符报错 |
| `tests/test_retrieval.py` | 12 | RRF 计分公式、双路命中优先、跨路去重、单路为空、同分稳定排序、**`fuse_rrf_multi` 多路（>2）累加与去重** |
| `tests/test_pipeline.py` | 19 | 门控拒答与放行两条路径、拒答时不调用生成模型、栏目透传、`final_n`/`rerank_top_n` 截断、父块去重、引用顺序与去重、悬空引用记录、debug 详略开关、空问题报错、栏目展示名透传与兜底、引用带文章标识、**多 Query 聚合取 max、单路失败丢路、门控用聚合分、展开失败降级、开关关闭不调 LLM** |
| `tests/test_multi_query.py` | 13 | 展开解析三级回退（JSON / 逐行 / 乱码）、原 query 恒在首位、去重保序、数量截断、关闭时不调 LLM、LLM 抛错回退 `[原 query]` |
| `tests/test_eval.py` | 5 | 检索层走共享 `retrieve()` 且 `sections` 恒 None、`expect_rejected` 库外样本跳过命中统计、门控放行触发 `EvalError` 列出 id、拒答通过、报告含拒答断言段 |
| `tests/test_api.py` | 37 | 元信息字段与**按站点分组的栏目结构**、静态页与资源可访问、未知路径 404、`AnswerResult` → JSON 的字段映射（含文章标识）、`sections`/`debug` 透传、空栏目转全库、拒答透传、空/缺/超长问题的中文校验提示、领域错误 502，响应与**日志**中的密钥擦除；图片端点的合法 png/webp、**任意目录深度下的图片**、六类拒绝路径、四类穿越形态（含 URL 编码与符号链接）、失败响应不泄漏服务器路径 |
| `tests/test_frontend_js.py` | 46 | `escapeHtml` 五字符、对抗性输入不产生可执行标记（script 标签 / `onerror` 属性 / 属性逃逸 / 伪协议链接）、表格（结构/对齐/单元格内标记/负例/代码块内不渲染）、链接（协议白名单、相对路径补全、与引用编号互不误伤、地址含括号）、删除线、嵌套列表、图片（有解析器出 `img` / 无解析器降级）、**引用图片地址按文章目录解析（两层站点与深层站点各一例）**、markdown 子集渲染、空输入安全、中英交界空格（需要 node） |

## 功能覆盖对照

| 需求 | 覆盖它的测试 | 类型 |
|------|-------------|------|
| F1 语料加载与父子切割 | `test_corpus.py`、`test_chunking.py` 全覆盖 | 单元 |
| F2 向量化与索引构建 | `test_index_store.py::test_build_*`（数量/维度/空输入校验） | 单元（真实向量化由 eval 覆盖） |
| F3 混合检索 | `test_index_store.py`（两路通道 + 栏目过滤）、`test_retrieval.py`（融合） | 单元 |
| F4 重排 | `test_pipeline.py`（`rerank_top_n` 截断、重排分数进 debug） | 单元（真实重排效果由 eval 覆盖） |
| F5 答案生成与门控 | `test_pipeline.py::test_gate_*`、`test_empty_retrieval_is_rejected` | 单元 |
| F6 引用溯源 | `test_pipeline.py::test_citation_*`、`test_dangling_citation_is_recorded` | 单元 |
| F7 Web 界面 | `test_api.py`（接口与静态页）、`test_frontend_js.py`（渲染与转义）；视觉与交互见「已知边界」 | 单元 + 手动 |
| F8 配置与密钥管理 | `test_config.py` 全覆盖（含 `test_redacted_hides_keys`） | 单元 |
| F9 索引管理入口 | 无单元测试，见「已知边界」 | 手动 |
| N1 数据本地性 | 无自动化，靠代码审查（仅 config 中的两个 base_url 出网） | 人工 |
| N2 密钥安全 | `test_config.py::test_redacted_hides_keys`、`test_api.py::test_ask_failure_redacts_keys_*` + 全项目密钥形态扫描 | 单元 + 人工 |
| N3 结果可复现 | `test_retrieval.py::test_ties_are_deterministic` + 双次构建对比 | 单元 + 人工 |
| N5 失败可诊断 | 各模块的报错用例（配置缺失、语料不符、索引漂移、维度不符） | 单元 |

### 模块 02（前端与接口）

| 需求 | 覆盖它的测试 | 类型 |
|------|-------------|------|
| F1 问答接口 | `test_api.py`（字段映射、`sections`/`debug` 透传、拒答透传、502 错误） | 单元 |
| F2 元信息接口 | `test_api.py::test_meta_*` | 单元 |
| F3 界面资源 | `test_api.py::test_index_page_is_served`、`test_static_assets_are_served` | 单元 |
| F4 提问与筛选 | `test_api.py::test_ask_forwarded_*`、`test_ask_empty_sections_means_whole_corpus` | 单元（界面交互由手动走查覆盖） |
| F5 加载态 | 无自动化，见「已知边界」 | 手动 |
| F6 拒答态区分 | `test_api.py::test_ask_passes_rejection_through`（数据面）；样式区分由手动走查覆盖 | 单元 + 手动 |
| F7 引用展示 | `test_frontend_js.py::test_citation_becomes_anchor`、`test_citation_inside_code_is_not_linkified` | 单元（折叠交互由手动走查覆盖） |
| F8 检索细节按需 | `test_pipeline.py::test_debug_detail_toggle`（载荷详略）、`test_api.py::test_ask_defaults_are_off`（默认不发） | 单元 |
| F9 输入与错误防护 | `test_api.py::test_empty_question_is_readable_422` 等三条、`test_frontend_js.py` 全部转义用例 | 单元 |
| F10 启动方式不变 | 无单元测试，见「已知边界」 | 手动 |
| N2 密钥隔离 | `test_api.py::test_meta_carries_no_key_material`、`test_ask_failure_redacts_keys_in_response`、`test_ask_failure_redacts_keys_in_logs` | 单元 |
| N3 渲染安全 | `test_frontend_js.py` 的四个对抗性输入用例（含 `assert_no_raw_markup` 不变量） | 单元 |
| N4 无外部资源 | 无自动化，靠 `grep -n "http(s)://" static/*` 与浏览器网络面板 | 人工 |
| N1 行为等价 | 无自动化（`pipeline.py` 零改动是结构性保证）；接受验收时以直调 `pipeline.ask` 比对 | 人工 |

### 模块 03（渲染扩展与知识库内置）

| 需求 | 覆盖它的测试 | 类型 |
|------|-------------|------|
| F3 图片资源读取 | `test_api.py::test_asset_serves_png`、`test_asset_serves_webp_with_explicit_media_type` | 单元 |
| F4 图片访问受约束 | `test_api.py::test_asset_requests_are_rejected`（六类）、`test_resolve_asset_rejects_directly`（四类穿越 + 符号链接）、`test_asset_rejection_leaks_no_server_path` | 单元 |
| F5 引用原文走渲染 | `test_frontend_js.py` 全部行内与块级用例（与答案共用同一条链路） | 单元（视觉由手动走查覆盖） |
| F6 表格渲染 | `test_table_renders_structure`、`test_table_alignment`、`test_table_cell_inline_markup`、两条负例 | 单元 |
| F7 行内语法补全 | `test_http_link_renders_anchor`、`test_relative_link_absolutised_with_resolver`、`test_strikethrough`、`test_nested_list` 等 | 单元 |
| F8 引用原文中的图片 | `test_image_renders_with_resolver`、`test_image_without_resolver_degrades_to_text` | 单元（真实加载由手动走查覆盖） |
| F9 折叠与新内容形态兼容 | 无自动化（CSS 行为），见「已知边界」 | 手动 |
| F10 答案正文图片降级 | `test_image_absolute_url_is_not_served`、`test_image_without_resolver_degrades_to_text` | 单元 |
| F11 配置指向项目内副本 | `test_config.py`（`kb_dir` 解析）+ 验收时 `status` 实测 | 单元 + 人工 |
| N1 路径安全 | `test_resolve_asset_rejects_directly`（含 `..` 与符号链接两类） | 单元 |
| N2 只读 | 无自动化，代码审查（端点只读 `FileResponse`，无写路径） | 人工 |
| N3 无外部请求 | 无自动化，靠 `grep -n "http(s)://" static/*` 与浏览器网络面板 | 人工 |
| N4 既有行为不回退 | 模块 01/02 全部既有测试仍全绿 | 单元 + 人工 |
| N6 仓库体积可控 | `grep -n "^kb/" .gitignore` | 人工 |

### 模块 05（多站点知识库与增量复用）

| 需求 | 覆盖它的测试 | 类型 |
|------|-------------|------|
| F1 新索引格式解析 | `test_corpus.py::test_load_all_pages`、`test_manifest_without_sites_is_reported` | 单元 |
| F2 变深目录布局 | `test_corpus.py::test_nested_directory_page_is_discovered`、`test_markdown_under_assets_is_not_corpus` | 单元 |
| F3 站点归属 | `test_corpus.py::test_page_fields`、`test_index_store.py`（站点点属性） | 单元 |
| F4 两级检索范围 | `test_pipeline.py::test_sections_forwarded_to_both_channels`（复合键透传到两路）；按站点筛选 = 传该站点全部栏目键，由前端组合 | 单元 + 手动 |
| F5 展示名来自索引文件 | `test_corpus.py::test_load_sites_reports_labels_and_order`、`test_api.py::test_meta_reports_counts_and_sections`、`test_pipeline.py::test_section_label_falls_back_to_last_segment` | 单元 |
| F6 增量构建缓存 | `test_reuse.py::test_same_content_hits_contexts_and_vectors`、`test_contextual.py::test_enrich_reuses_notes_without_calling_model` | 单元（真实命中数由构建日志见证） |
| F7 缓存自失效 | `test_reuse.py::test_chat_model_change_disables_context_reuse`、`test_embedding_model_change_disables_vector_reuse`、`test_chunking_change_invalidates_contexts` | 单元 |
| F8 缓存播种 | 复用来源即上一版索引，`test_reuse.py` 全部用例都在验证这条路径 | 单元 |
| F9 构建统计可见 | `test_contextual.py::test_enrich_reuses_notes_without_calling_model`（记账）；输出本身靠构建日志目视 | 单元 + 人工 |
| F10 缓存降级 | `test_reuse.py::test_missing_index_degrades_to_empty`、`test_corrupted_entries_degrades_to_empty`、`test_vector_count_mismatch_degrades` | 单元 |
| F11 图片访问适配深层布局 | `test_api.py::test_asset_serves_image_under_deep_path` | 单元 |
| F12 图片访问约束不变 | `test_api.py` 六类拒绝 + 四类穿越用例 | 单元 |
| F13 前端图片地址按文章目录解析 | `test_frontend_js.py::test_citation_image_url_for_two_level_site`、`test_citation_image_url_for_deep_site` | 单元 |
| F14 前端筛选分组呈现 | 无自动化（DOM 结构 + CSS），见「已知边界」 | 手动 |
| F15 知识面表述一致 | 无自动化（提示词文本）；靠真实提问走查 | 手动 |
| F16 评测集更新 | `eval.py::load_eval_set` 的字段校验 + 验收时 `eval` 实跑 | 单元 + 人工 |
| F17 知识库副本更新 | 无自动化，拷贝后比对文件清单与抽查内容 | 人工 |
| N1 密钥不泄漏 | `test_config.py::test_redacted_hides_keys`、`test_api.py::test_meta_carries_no_key_material` | 单元 |
| N2 路径安全 | `test_resolve_asset_rejects_directly` | 单元 |
| N6 缓存不误用 | `test_reuse.py` 的四条自失效用例 | 单元 |
| N7 仓库体积可控 | `grep -n "^kb/\|^data/" .gitignore` | 人工 |

### 模块 06（多 Query 扩展与评估口径）

| 需求 | 覆盖它的测试 | 类型 |
|------|-------------|------|
| F1 展开（原 query 恒在首位） | `test_multi_query.py::test_original_query_always_first` 等 | 单元 |
| F2 多路检索 | `test_pipeline.py::test_multi_rerank_aggregates_max_per_query`（多路召回 + 每 query 命中明细） | 单元 |
| F3 RRF 泛化融合 | `test_retrieval.py::test_multi_three_paths_accumulate_scores`、`test_multi_dedupes_across_more_than_two_paths` | 单元 |
| F4 精排对齐（聚焦打分 + max 聚合） | `test_pipeline.py::test_multi_rerank_aggregates_max_per_query`（聚合取 max、原 query 不参与打分） | 单元 |
| F5 可观测 debug | `test_debug_detail_toggle` + 多路用例断言 `queries`/`expanded`/`per_query` | 单元 |
| F6 开关与容错 | `test_multi_disabled_does_not_call_llm`、`test_multi_degrades_to_original_when_expansion_fails`、`test_multi_single_route_failure_skips_only_that_route` | 单元 |
| F7 裁判启用标准答案 | 用户裁为「本轮不做」（见项目根 `README.md` 与归档 README） | 已裁掉 |
| F8 全库口径评估 | `test_eval.py::test_retrieval_uses_whole_corpus_sections_none` | 单元（真实指标由 `eval --no-judge` 承担） |
| F9 全量判分 + 独立裁判 | 用户裁为「本轮不做」 | 已裁掉 |
| F10 库外拒答断言 | `test_eval.py::test_run_eval_raises_when_out_of_kb_passes`、`test_run_eval_passes_when_out_of_kb_rejected` | 单元（真实断言由 `eval --no-judge` 承担） |
| F11 检索链路复用 | `test_eval.py::test_retrieval_uses_whole_corpus_sections_none`（评估走与在线同编排）、`test_pipeline.py` 多路用例 | 单元 |

### 模块 06「不做」的判定依据

| 决策 | 依据 |
|------|------|
| 生成层裁判（F7/F9）本轮不做 | 用户明确选择；`judge_model` 仍与 `chat_model` 相同（`deepseek-v4.1-flash`），`judge_answers`/`JUDGE_*`/`--judge-sample`/`--no-judge` 零改动 |
| 不改语料、不重建索引 | 硬约束：「不自创知识库内容」；纯查询编排层改动 |

## 模块 ↔ 测试 双向索引

| 模块（源码） | 对应测试文件 | 验证什么 |
|--------------|-------------|----------|
| `config.py` | `tests/test_config.py` | 配置与密钥的加载、校验、打码、网关额外请求头 |
| `corpus.py` | `tests/test_corpus.py` | 新索引格式解析、递归发现、与磁盘集合对账、展示名与顺序 |
| `reuse.py` | `tests/test_reuse.py` | 内容寻址的命中与失效、模型变更的显式把关、降级为空源 |
| `chunking.py` | `tests/test_chunking.py` | 父子切分的边界、原子块保护、尺寸约束 |
| `contextual.py` | `tests/test_contextual.py` | 背景说明解析与截断、失败降级、开关与拆批、复用命中跳过调用 |
| `index_store.py` | `tests/test_index_store.py` | 索引构建/存取/对齐校验、两条检索通道、站点点属性 |
| `retrieval.py` | `tests/test_retrieval.py` | RRF 融合的正确性与稳定性（含 `fuse_rrf_multi` 多路） |
| `multi_query.py` | `tests/test_multi_query.py` | 展开解析三级回退、原 query 恒在首位、去重截断、关闭与失败降级、低温透传 |
| `pipeline.py` | `tests/test_pipeline.py` | 门控、引用解析、编排参数透传、展示名与文章标识、多 Query 聚合/降级/开关 |
| `eval.py` | `tests/test_eval.py` | 检索层走共享 `retrieve()`、sections 恒 None、库外断言（放行红 / 拒答过 / 报告段） |
| `embedder.py` / `reranker.py` / `llm.py` | 无（外部 API 客户端） | 见「已知边界」 |
| `api.py` | `tests/test_api.py` | 路由与静态托管、图片端点与路径安全、请求校验、字段映射、错误与密钥擦除 |
| `static/app.js` | `tests/test_frontend_js.py` | 转义不变量、markdown 子集渲染（表格/列表/链接/删除线/图片）、引用锚点与图片地址（需 node） |
| `static/index.html` / `static/styles.css` | 无（结构与样式） | 见「已知边界」 |
| `eval.py` | `tests/test_eval.py` | 检索层接入、库外断言、报告段 |
| `cli.py` | 无（入口与编排） | 见「已知边界」 |

## 已知边界 / 存量差异处置

本仓库**无存量失败用例**，全部 213 条稳定全绿。

以下部分**没有单元测试**，属有意为之，不是遗漏：

1. **外部 API 客户端**（`embedder.py` / `reranker.py` / `llm.py`）——测它们等于测网络。客户端内的重试、批处理、错误翻译逻辑靠代码审查与一次性的真实调用验证；错误文案不泄漏密钥这一点由 `test_config.py::test_redacted_hides_keys` 与全项目密钥形态扫描共同守住。
2. **`contextual.py` 生成内容的质量**——背景说明「写得好不好」是 LLM 输出的语义质量，断言其内容等于把测试绑死在模型行为上。测试覆盖的是**解析与降级契约**；「背景是否与原文相关」靠构建时的完成计数与人工抽样。
3. **`cli.py`**——入口与编排，验证方式是实际跑一遍 `build → status → serve → 提问`，见项目 `README.md`。
4. **`eval.py` 的评估语义**——「检索层接入共享 `retrieve()`、库外断言、报告渲染」已有单测（`test_eval.py`）；但「指标标定得对不对」（比如换一个更弱的 Embedding 配置后 Hit@K/MRR 应可观测下降）仍是语义验证，靠真实跑 `python -m xiaolinrag eval`。
5. **界面在真实浏览器里的视觉与交互**（加载态、拒答样式、引用折叠与展开、**按站点分组的筛选排版**、窄屏布局、错误行内提示）——JSDOM 类环境测不出排版，截图又无法断言像素。做法是：可判定为纯函数的部分抽出来交给 `test_frontend_js.py`（转义、markdown 渲染、引用图片地址），其余在验收时用 headless chromium 截图 + 逐项手动走查，走查结论记录在当轮 `checklist.md`。**启动方式（`serve` 阻塞进程）同样归入这一类。**
6. **`static/index.html` / `static/styles.css`**——结构与样式，无逻辑可断言；`index.html` 的钩子 id 与 `app.js` 的一致性靠目视核对，样式中的禁用项（无装饰性背景渐变、无重阴影、无外链字体）靠 grep 检查。
7. **知识库副本 `kb/`**——它是数据不是代码，没有「单元测试」这一说。完整性靠拷贝后与源目录逐文件（路径 + 大小）比对、再抽查若干文件的 md5；副本与原库**不自动同步**，原库更新后需重新拷贝。
8. **依赖 CSS 与真实浏览器的行为**——折叠边缘的遮罩渐隐、表格在窄屏下的换行、图片在卡片内的自适应尺寸，都属于「排版对不对」，断言像素没有意义。做法是起真实服务后用 headless chromium 截图逐项目视，结论记在当轮 checklist。图片是否真的加载出来也归这一类：`<img>` 标签存在不等于图片渲染成功，要同时看截图与「失败降级文案是否出现」。
9. **增量复用的实际收益（模块 05）**——单元测试能证明「命中时不调用模型」，但「这次构建到底省了多少」只能由构建日志见证。另外**复用源必须来自上一版索引**：把 `data/index/` 删掉再构建就是全量，这与「缓存文件丢了」是同一件事，不是故障。
10. **svg 不提供（模块 05）**——全库仅 1 张 svg，按白名单设计一律 404，前端降级为可读文本。这条边界由 `test_api.py::test_asset_requests_are_rejected` 里的 svg 用例钉住，属有意为之。
11. **对话网关的额外请求头（模块 05）**——当前网关要求 `x-opencode-session`（客户端自行生成的会话标识，不参与鉴权），由 `config.yaml` 的 `models.chat_extra_headers` 传入。这条无法用单元测试覆盖（测它等于测网关），靠一次真实调用验证。
12. **门控的「库外重叠」边界（模块 06）**——重排器度量内容重叠，库外问题若恰好撞上库内重头戏（实测「React 调度器」对库内大片「调度」内容在原 query 下仍打到 0.42、聚焦聚合 0.83），门控可能放行；此时由生成层诚实兜底（答「未找到」不编造）。这是**真实链路实测**得到的边界，mock 测不出（mock 的重排分是内定的），记录在本轮 `checklist.md` 与 `docs/06-多Query扩展.md` §4-6，不当成恒定断言。换机制（库域判别分类器）是后续候选，本轮不做（用户确认）。

**跳过不等于通过**：`tests/test_frontend_js.py` 在未安装 node 的环境会整份 skip。看到 skip 时其余测试依然全绿，但前端纯函数这一层实际未被验证——需先确认 node 可用。

**真实语料相关的数字**来自 2026-09-20 / 09-21 的两次实测（接入两个站点：235 篇 / 7053 子块 / 2215 父块；06 轮评测集扩到 **62 行 = 50 道 + 8 道高频域探测 + 4 道库外样本**），写入文档是为说明量级与取舍，不应被当作断言。语料或模型变化后需重跑 `python -m xiaolinrag build` 与 `python -m xiaolinrag eval` 复核。
