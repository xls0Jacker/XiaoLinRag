# 模块解读 06：多 Query 扩展

> 本文拆解 `xiaolinrag/multi_query.py`、`retrieval.py`（`fuse_rrf_multi`）、`pipeline.py`（`retrieve`/`rerank_aggregate`）与 `eval.py`（检索层接入与库外断言），
> 讲清「复合 query 如何被拆开检索、精排如何不被复合 query 注水、评估如何与在线同口径」这条链路。
> 全文基于本项目 `xiaolinrag/` 包（模块解读，供自研参考）。

## 1. 模块职责

| 文件 | 职责 |
|------|------|
| `multi_query.py` | Query 展开：LLM 把用户问题拆成多个角度、各成一 query；原 query 恒在首位；失败一律降级不阻断 |
| `retrieval.py`（`fuse_rrf_multi`） | 把「双路 RRF」泛化为「N query × 双通道」任意多路的倒数排名融合 |
| `pipeline.py`（`retrieve` / `rerank_aggregate`） | 共享检索编排：展开 → 多路召回 → RRF → 聚焦精排聚合（max）；在线 `ask` 与评估**共用这一个函数** |
| `eval.py` | 检索层切到共享 `retrieve()`（全库口径、无栏目预过滤）；库外样本的拒答断言 |

**背景**：两个真实线上失败同根——「三次握手+四次挥手」的问题里，挥手/断开内容（库内真实存在）进不了 top-5 生成上下文，模型回「未找到」；「Cookie vs Session」问题被广告块顶榜，真讲 HTTP Cookie 的块垫底。根因是复合 query 走单路检索、固定预算，检索前缺少「拆角度」。知识库自身方法论（`kb/xiaolinnote/rag` docs 12 方法四 / 13 第三路 / 14 Multi-Query）给出对应对策：**Query 预处理阶段的多 Query 扩展**。

## 2. 核心数据结构

### `RetrievalResult`（`pipeline.py:62`）

一次检索编排的产物，`ask` 与评估共同消费：

```python
@dataclass
class RetrievalResult:
    queries: list[str]        # 实际用于检索的 query（原 query 恒在首位）
    expanded: list[str]       # 额外展开的 query；空 = 单 query 模式
    fused: list[ScoredHit]    # RRF 融合后的全量候选（按 chunk_id 去重）
    ranked: list[ScoredHit]   # 精排后的降序列表（多路模式下聚合分在 rerank_score）
    per_query: list[dict]     # 每个 query 的命中明细，供 debug 区分（F5）
```

它为什么长这样：**一次检索的结果是「多 query × 双通道」的聚合**，单是「融合候选 + 精排结果」不够——评估要算精排前后的 Hit 差、debug 要能还原每个展开 query 各自捞到了什么，所以把沿途产物全部装进去。`ask()` 只消费 `fused`/`ranked`，评估只消费 `ranked`/`fused`，`per_query` 只进 debug。

### `MultiQueryConfig`（`config.py`）

```python
MultiQueryConfig(enabled: bool = True, num_queries: int = 4, temperature: float = 0.2)
```

`num_queries` 是**含原 query 的总数**：`expand_queries` 让 LLM 额外生成 `num_queries - 1` 条。关闭或数量为 1 时走单 query 链路，不与现状产生任何字节差（AC4 回归护栏）。

## 3. 关键流程

### 3.1 在线查询链路（`retrieve` → `ask`）

```
用户问题
   │
   ▼ 展开（multi_query.expand_queries）           原 query 恒在首位（召回不丢细节）
   ▼ 逐 query：向量 top20 + BM25 top20（2N 路）
   ▼ RRF 融合（fuse_rrf_multi）                   跨路按 chunk_id 去重、按排名累加
   ▼ 截取 rerank_top_n 个候选
   ├─ 多路：rerank_aggregate（用各展开 query 打分，按子块取 max）
   └─ 单路：reranker.rerank(原 query, …)          与旧链路逐字节一致
   ▼ 门控（聚合最高分 < 0.3 → 拒答）
   ▼ final_n 取 top → 父块去重 → 生成 → 引用解析
```

### 3.2 聚合精排 `rerank_aggregate`（F4，必要行为）

精排**不再对复合 query 打全部候选的分**，而是每个聚焦展开 query 各打一遍、按子块取各 query 中的最高分聚合排序：

```python
for query in scoring_queries:          # 各聚焦展开 query
    for hit in reranker.rerank(query, candidate_entries, cfg, top_n=len(candidate_entries)):
        if hit.rerank_score > best_score.get(key, -1.0):
            best_score[key] = score    # 按子块取 max
```

为什么这是必要行为而不是优化：复合 query 会给高分噪声注水——受控 probe 实测，同一条广告块在复合 query 下 0.996、在聚焦 query 下 0.002~0.04；真讲 Cookie 的块反而从 0.62~0.77 的噪声里露出来、登顶 0.90。**max 聚合语义与门控对齐**：任一聚焦角度认为相关内容足够即准入，广告块在所有聚焦 query 下都接近 0、自然降权，不需要单独的推广块过滤。

### 3.3 门控与库外拒答（`ask` / `evaluate_rejections`）

门控用**聚合后的最高分**对照 `gate_threshold=0.3`，低于即拒答。评估侧给每个标记 `expect_rejected` 的库外样本跑完整 `ask`，断言 `rejected=True`；任一放行即 `EvalError`（`run_eval`）。库外样本无目标页，**不进检索层命中统计**——它只回答问题「门控有没有把库外挡下来」。

## 4. 要点小结

1. **为什么展开结果必须保留原 query？**
   改写可能丢细节（docs 14 硬约束）。原 query 恒在首位是「召回端不丢细节」，但**不参与精排打分**——精排打分的是展开出来的聚焦 query，两者用途不同，别混。

2. **聚合为什么取 max 而不是平均或取其一？**
   取 max 语义是「任一聚焦角度认为相关即准入」，与门控对齐。广告块在所有聚焦 query 下都接近 0，max 不会捞它；真块在对应的那个 query 下登顶。若取平均，广告块会被其它观看角度的 0 拉平、真块反而被不相关的观看角度稀释。

3. **为什么评估检索层要走全库口径（sections 恒 None）？**
   跨栏目混淆是召回质量的真实组成部分——如果按栏目预过滤，那么「把 A 栏内容当成问题答案」这一类错误被静默排除出指标，Hit@K/MRR 虚高。评估测的就是线上链路，线上 `ask` 不预过滤，评估也不预过滤（F7）。

4. **为什么任一展开 query 检索失败只丢该路、而不上抛？**
   复合 query 拆出来的 N 路是平行意见，某一路模型/服务抖了不该让整个回答失败。单 query 模式仍照原样上抛（与现状一致），只有多路模式才宽容。

5. **库外拒答为什么能让评估直接失败？**
   简历上的「知识库外全部拒答」是个可证伪的承诺，不是顺嘴的描述。把库外样本写进评测集、让门控放行=评估红，这个数字才有口径（F8）。

6. **已知边界：库外问题撞上库内重头戏时分不清**
   重排器度量的是**内容重叠**。「React 调度器内部实现」这个库外问题，恰好在原 query 下都能打到 0.42~0.43、聚焦 query 下 0.66~0.83——因为库内正有大片「调度」内容（OS 调度 / ReAct / 任务调度）与之语义重叠。任何单一阈值都无法同时挡住这类「恰好重叠」的库外问题、又放行真实库内复合问题（Cookie 复合 query 在原 query 下仅 0.034）。实测已证（换向量库也不改变「重叠度」这个打分输入）。底线保障是**生成层诚实兜底**：即使门控放行，生成模型看到上下文里没有 React 内容，也会照系统提示答「知识库中未找到」，不编造。换机制（库域判别分类器）是后续候选，本轮不做。

## 5. 测试绑定

**对应的测试文件：**
- `tests/test_multi_query.py` —— 展开解析三级回退、原 query 恒在首位、去重截断、关闭/失败降级
- `tests/test_retrieval.py` —— `fuse_rrf_multi` 多路分数累加、跨 >2 路去重、空列表、同分稳定
- `tests/test_pipeline.py` —— 多路聚合取 max、单路失败丢路、门控用聚合分、降级、开关回归、AC3 原 query 门控负面记录
- `tests/test_eval.py` —— 检索层 sections 恒 None、库外样本跳过命中统计、放行触发 EvalError、拒答通过、报告断言段

**怎么验证本模块：**
- 整文件（全绿）：`python -m pytest tests/test_multi_query.py tests/test_retrieval.py tests/test_pipeline.py tests/test_eval.py -q`
- 单条用例：`python -m pytest tests/test_pipeline.py::test_multi_rerank_aggregates_max_per_query -q`

**需要知道：**
- `docs/testing.md` 是全部测试的入口与总表，可反查任意模块；无存量失败。
- 门控的边界（AC3 的 React 放行、答案诚实兜底）是**真实链路实测**，单元测试不覆盖——mock 测不出重排器的内容重叠行为；需真实 API 才能复现，结论记录在本轮 `checklist.md` 与第 4 节第 6 条。
- 真实链路数字（聚合分、门控反应分数）依赖重排模型，语料或模型变化后需重跑 `python -m xiaolinrag eval` 复核。