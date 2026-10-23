# ATTUNE: A Context-Aware Preference Gating Engine for Technical Explanations
## Technical & Research Specification Document

- **Document Version:** 1.0.0
- **Status:** Approved / Base Contract
- **Target Implementation:** ATTUNE Engine Core (`attune-core`)
- **Repository:** `d:/PROJECT_SOUCEFILE/attune`

---

## Table of Contents
1. [Core Research Question & Scope](#1-core-research-question--scope)
   - [1.1 Primary Research Question](#11-primary-research-question)
   - [1.2 Target Application Domain](#12-target-application-domain)
   - [1.3 Non-Goals & Boundary Conditions](#13-non-goals--boundary-conditions)
2. [The 4-Dimension Preference Taxonomy](#2-the-4-dimension-preference-taxonomy)
   - [2.1 Taxonomy Architecture](#21-taxonomy-architecture)
   - [2.2 Dimension 1: Depth](#22-dimension-1-depth)
   - [2.3 Dimension 2: Pedagogy](#23-dimension-2-pedagogy)
   - [2.4 Dimension 3: Illustration](#24-dimension-3-illustration)
   - [2.5 Dimension 4: Structure](#25-dimension-4-structure)
   - [2.6 Canonical Feature Representation](#26-canonical-feature-representation)
3. [Data Schema & Gating Decision Tuple](#3-data-schema--gating-decision-tuple)
   - [3.1 Decision Tuple Definition](#31-decision-tuple-definition)
   - [3.2 JSON Schema Specification](#32-json-schema-specification)
   - [3.3 Concrete Instance Examples](#33-concrete-instance-examples)
4. [Decision & Priority Rules](#4-decision--priority-rules)
   - [4.1 The Golden Priority Rule (Query-Level Hard Overrides)](#41-the-golden-priority-rule-query-level-hard-overrides)
   - [4.2 Label 1 (Apply) Criteria](#42-label-1-apply-criteria)
   - [4.3 Label 0 (Suppress) Criteria](#43-label-0-suppress-criteria)
   - [4.4 Hybrid Arbitration Pipeline & Flow](#44-hybrid-arbitration-pipeline--flow)
5. [Mathematical Definitions of Metrics](#5-mathematical-definitions-of-metrics)
   - [5.1 Core Classification Metrics](#51-core-classification-metrics)
   - [5.2 Negative Transfer Rate (NTR)](#52-negative-transfer-rate-ntr)
   - [5.3 False Suppression Rate (FSR)](#53-false-suppression-rate-fsr)
   - [5.4 Repeated-Correction Rate (RCR)](#54-repeated-correction-rate-rcr)
   - [5.5 Adaptation Latency (AL)](#55-adaptation-latency-al)
   - [5.6 Expected Calibration Error (ECE)](#56-expected-calibration-error-ece)
6. [Evaluation Protocol & Baselines](#6-evaluation-protocol--baselines)
   - [6.1 Comparative Benchmark Configurations](#61-comparative-benchmark-configurations)
   - [6.2 Benchmark Dataset Splits & Stratification](#62-benchmark-dataset-splits--stratification)
   - [6.3 Graduation Criteria](#63-graduation-criteria)
   - [6.4 Statistical Significance Protocol](#64-statistical-significance-protocol)
   - [6.5 External Academic Benchmark Generalization](#65-external-academic-benchmark-generalization)

---

## 1. Core Research Question & Scope

### 1.1 Primary Research Question
> **Can a lightweight trained preference-gating model use interaction evidence to decide when a learned preference should influence the current answer, reducing inappropriate personalization (Negative Transfer) without losing useful personalization?**

Personalized language assistants frequently suffer from **over-generalization** and **preference leakage**:
- When a user requests an in-depth mathematical derivation while studying quicksort proofs, conventional memory architectures append a static trait (e.g., `Style: Formal / Mathematical Proof`) to subsequent system instructions.
- If that same user later queries a rapid bash command syntax or a production debugging flag, the model inappropriately delivers verbose proofs or academic treatises.
- Conversely, naive sliding windows or pure retrieval methods often purge or retrieve mismatched historical traces, leading to **under-personalization** (suppressing preferences in legitimate recurring learning scenarios).

ATTUNE introduces an explicit, lightweight gating decision layer that sits between conversation episodic memory and answer generation. The model predicts the contextual applicability of candidate preferences given conversational state, recent user feedback, and semantic domain scope.

### 1.2 Target Application Domain
Personalized technical explanations in **Computer Science**, restricted to four core subdomains:
1. **Algorithms & Data Structures (`algorithms_learning`)**: Asymptotic complexity analysis, algorithmic invariants, tree/graph traversal, dynamic programming mechanics, and recurrence relations.
2. **Systems & Architecture (`system_tradeoffs`)**: Operating systems, concurrency primitives, memory hierarchy, cache coherence, distributed consistency models, and network protocol internals.
3. **Database & Tools (`database_tools`)**: Query execution plans, index selection, ACID properties, CI/CD pipeline automation, CLI diagnostic tools, and container runtimes.
4. **Language Syntax & Quick Reference (`syntax_lookup`)**: Idiomatic syntax, standard library lookup, API signatures, pattern-matching constructs, and rapid compiler error triage.

### 1.3 Non-Goals & Boundary Conditions
To maintain experimental rigor and unambiguous research validation, the following areas are strictly **out of scope**:
- **Training Full Foundation LLMs**: ATTUNE does not pretrain, fine-tune, or modify the weights of the foundational generative LLM (e.g., Llama-3, Mistral, GPT-4). Personalization is controlled via contextual gating and prompt conditioning.
- **Open-Ended Social Chatbots**: ATTUNE does not handle conversational roleplay, persona synthesis, emotional support, or casual chit-chat.
- **Open-Domain User Personality & Psychometrics**: ATTUNE does not infer or persist open-ended psychological profiles, Big Five traits, demographic markers, or emotional dispositions.
- **Cross-User Collaborative Filtering**: Gating decisions are computed strictly within a self-contained user session (`episode_id`) and its explicit evidence trail, without cross-user profile aggregation.

---

## 2. The 4-Dimension Preference Taxonomy

The ATTUNE engine restricts personalized style manipulation to an orthogonal, 4-dimension discrete taxonomy. Each dimension consists of mutually exclusive discrete poles, represented as formal enumeration tokens.

```
+-----------------------------------------------------------------------------------+
|                        ATTUNE 4-DIMENSION PREFERENCE TAXONOMY                     |
+-------------------+-----------------------------------+---------------------------+
| Dimension         | Token Key                         | Discrete Values           |
+-------------------+-----------------------------------+---------------------------+
| 1. Depth          | DEPTH                             | CONCISE | DETAILED        |
| 2. Pedagogy       | PEDAGOGY                          | INTUITIVE | FORMAL        |
| 3. Illustration   | ILLUSTRATION                      | WORKED_TRACE | CONCEPTUAL |
| 4. Structure      | STRUCTURE                         | MODULAR | CONTINUOUS      |
+-------------------+-----------------------------------+---------------------------+
```

### 2.1 Taxonomy Architecture
Every preference instance tracked by the engine must bind to exactly one `(dimension, value)` pair. Composite styles are formed strictly as a set of active dimension values.

### 2.2 Dimension 1: Depth
Controls the semantic density and volumetric coverage of the response.

| Discrete Token | Canonical Definition | Concrete Manifestation | Negative Trigger Context |
|---|---|---|---|
| `DEPTH:CONCISE` | Minimalist, high-density response stripping non-essential context; focuses exclusively on the immediate solution or answer. | ≤ 3 sentences or minimal direct code snippet without explanatory preamble. | In-depth architectural review or conceptual study session where foundational details are requested. |
| `DEPTH:DETAILED` | Comprehensive coverage including edge cases, boundary behavior, time/space trade-offs, and failure modes. | Multi-section elaboration analyzing runtime, memory footprint, and architectural alternatives. | Rapid syntax lookup or emergency terminal fix where immediate action is required. |

### 2.3 Dimension 2: Pedagogy
Controls the cognitive approach and epistemological framing used to explain the topic.

| Discrete Token | Canonical Definition | Concrete Manifestation | Negative Trigger Context |
|---|---|---|---|
| `PEDAGOGY:INTUITIVE` | Reasoning through metaphors, real-world analogies, high-level operational models, and mental shortcuts. | Compares cache invalidation to updating a physical warehouse inventory whiteboard. | Strict formal verification, academic homework, or exact algorithmic proof. |
| `PEDAGOGY:FORMAL` | Axiomatic derivations, mathematical invariants, inductive proofs, and rigorous symbolic definitions. | Proves loop termination using well-founded relations or asymptotic bounds via Master Theorem. | Pragmatic CLI debugging or rapid API configuration where formalism impedes utility. |

### 2.4 Dimension 3: Illustration
Controls the presence and execution style of concrete execution artifacts.

| Discrete Token | Canonical Definition | Concrete Manifestation | Negative Trigger Context |
|---|---|---|---|
| `ILLUSTRATION:WORKED_TRACE` | Step-by-step state machine execution, variable evolution tables, pointer updates, or stack frame traces. | Manual table showing index values `i`, `j`, `pivot`, and array partitions for each partition step. | High-level system design discussion where low-level variable mutation is irrelevant. |
| `ILLUSTRATION:CONCEPTUAL` | Direct conceptual mechanics and abstract state transitions without line-by-line register/variable traces. | Explains Paxos quorum intersection conceptually without running a mock 5-node sequence log. | Student debugging an off-by-one pointer error in an AVL rotation. |

### 2.5 Dimension 4: Structure
Controls visual organization and typographical layout.

| Discrete Token | Canonical Definition | Concrete Manifestation | Negative Trigger Context |
|---|---|---|---|
| `STRUCTURE:MODULAR` | Information organized in discrete numbered stages, atomic bullet points, key-value summaries, and callouts. | Strict use of Markdown headings, bulleted prerequisites, complexity blocks, and code panels. | Request for continuous essay explanations, patent disclosures, or seamless prose narratives. |
| `STRUCTURE:CONTINUOUS` | Narrative prose flowing logically through connected paragraphs without fragmenting into checklists. | Synthesized academic prose connecting trade-offs in narrative transitions. | Rapid reference manuals or cheat sheets requiring scannability. |

### 2.6 Canonical Feature Representation
In the model input representation, preferences are encoded as normalized categorical tokens:
```json
"candidate_preference": "PEDAGOGY:FORMAL"
```

---

## 3. Data Schema & Gating Decision Tuple

Every training example, validation point, and inference runtime evaluation payload is encapsulated as a serialized JSON object adhering to the schema below.

### 3.1 Decision Tuple Definition
A Gating Decision Tuple $T$ is defined as:
$$T = \langle \text{episode\_id}, \text{current\_query}, \text{candidate\_preference}, \text{preference\_scope}, \mathcal{E}, \mathcal{O}, y, \mathcal{S} \rangle$$
Where:
- $\text{episode\_id} \in \Sigma^+$: Unique session identifier.
- $\text{current\_query} \in \Sigma^+$: Current user technical input.
- $\text{candidate\_preference} \in \mathcal{P}$: Element of the 4-dimension taxonomy.
- $\text{preference\_scope} \in \mathcal{D}$: Domain classification where preference was inferred.
- $\mathcal{E} = [e_1, e_2, \dots, e_k]$: Prior interaction quotes serving as supporting evidence.
- $\mathcal{O} \in \Sigma^+ \cup \{\text{null}\}$: Subsequent explicit contradictory statement if present.
- $y \in \{0, 1\}$: Binary ground-truth gate decision (`gold_applicability`).
- $\mathcal{S} \in \{\text{same\_context}, \text{context\_shift}, \text{explicit\_reversal}, \text{ambiguous\_neutral}\}$: Scenario taxonomy classification.

### 3.2 JSON Schema Specification

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "AttuneGatingDecisionTuple",
  "type": "object",
  "required": [
    "episode_id",
    "current_query",
    "candidate_preference",
    "preference_scope",
    "supporting_evidence",
    "recent_override",
    "gold_applicability",
    "scenario_type"
  ],
  "properties": {
    "episode_id": {
      "type": "string",
      "pattern": "^ep_[a-zA-Z0-9_-]+$",
      "description": "Unique session identifier grouping consecutive interaction turns."
    },
    "current_query": {
      "type": "string",
      "minLength": 3,
      "description": "The exact verbatim text of the user's active technical query."
    },
    "candidate_preference": {
      "type": "string",
      "enum": [
        "DEPTH:CONCISE",
        "DEPTH:DETAILED",
        "PEDAGOGY:INTUITIVE",
        "PEDAGOGY:FORMAL",
        "ILLUSTRATION:WORKED_TRACE",
        "ILLUSTRATION:CONCEPTUAL",
        "STRUCTURE:MODULAR",
        "STRUCTURE:CONTINUOUS"
      ],
      "description": "Canonical token indicating which preference is under evaluation."
    },
    "preference_scope": {
      "type": "string",
      "enum": [
        "algorithms_learning",
        "system_tradeoffs",
        "database_tools",
        "syntax_lookup"
      ],
      "description": "Technical domain where the preference evidence originated."
    },
    "supporting_evidence": {
      "type": "array",
      "items": {
        "type": "string",
        "minLength": 1
      },
      "minItems": 0,
      "description": "Verbatim textual evidence from prior conversation turns."
    },
    "recent_override": {
      "type": ["string", "null"],
      "description": "Direct subsequent feedback reversing or modifying the preference, or null if uncontradicted."
    },
    "gold_applicability": {
      "type": "integer",
      "enum": [0, 1],
      "description": "Binary ground-truth: 1 if preference must be applied; 0 if suppressed."
    },
    "scenario_type": {
      "type": "string",
      "enum": [
        "same_context",
        "context_shift",
        "explicit_reversal",
        "ambiguous_neutral"
      ],
      "description": "Experimental cohort category evaluating specific failure modes."
    }
  },
  "additionalProperties": false
}
```

### 3.3 Concrete Instance Examples

#### Example 1: `same_context` (Applicability = 1)
```json
{
  "episode_id": "ep_cs_algo_0841",
  "current_query": "Explain how the Kosaraju-Sharir algorithm finds strongly connected components and why two DFS passes are required.",
  "candidate_preference": "PEDAGOGY:FORMAL",
  "preference_scope": "algorithms_learning",
  "supporting_evidence": [
    "User turn 2: 'Please skip high-level intuitions and give me formal loop invariants or topological ordering proofs for graph algorithms.'"
  ],
  "recent_override": null,
  "gold_applicability": 1,
  "scenario_type": "same_context"
}
```

#### Example 2: `context_shift` (Applicability = 0)
```json
{
  "episode_id": "ep_cs_algo_0841",
  "current_query": "What is the tar command flag to extract a .tar.gz archive directly to /opt/tools without preserving ownership?",
  "candidate_preference": "PEDAGOGY:FORMAL",
  "preference_scope": "algorithms_learning",
  "supporting_evidence": [
    "User turn 2: 'Please skip high-level intuitions and give me formal loop invariants or topological ordering proofs for graph algorithms.'"
  ],
  "recent_override": null,
  "gold_applicability": 0,
  "scenario_type": "context_shift"
}
```

#### Example 3: `explicit_reversal` (Applicability = 0)
```json
{
  "episode_id": "ep_sys_db_1029",
  "current_query": "Compare Raft leader election with Paxos Phase 1.",
  "candidate_preference": "DEPTH:DETAILED",
  "preference_scope": "system_tradeoffs",
  "supporting_evidence": [
    "User turn 1: 'Give me exhaustive breakdowns of distributed protocols including edge-case network partitions.'"
  ],
  "recent_override": "User turn 4: 'These explanations are way too long. Going forward give me concise high-level summaries only.'",
  "gold_applicability": 0,
  "scenario_type": "explicit_reversal"
}
```

#### Example 4: `ambiguous_neutral` (Applicability = 0)
```json
{
  "episode_id": "ep_syntax_4492",
  "current_query": "Show an example of std::variant visitor pattern in C++20.",
  "candidate_preference": "ILLUSTRATION:WORKED_TRACE",
  "preference_scope": "syntax_lookup",
  "supporting_evidence": [
    "User turn 1: 'Okay, I see how that works.'"
  ],
  "recent_override": null,
  "gold_applicability": 0,
  "scenario_type": "ambiguous_neutral"
}
```

---

## 4. Decision & Priority Rules

The ATTUNE engine executes a two-tier decision hierarchy:
1. **Tier 1**: Deterministic Rule-Based Pre-Filter (The Golden Priority Rule & Hard Negatives).
2. **Tier 2**: Learned Gating Classifier (Contextual Alignment & Scope Coherence).

```
                      +-----------------------------+
                      |   Current Query Received    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Tier 1: Golden Rule Engine  |
                      | Does query contain an       |
                      | explicit style directive?   |
                      +--------------+--------------+
                                     |
                         +-----------+-----------+
                         |                       |
                      [YES]                     [NO]
                         |                       |
                         v                       v
               +--------------------+  +--------------------+
               | Hard Override:     |  | Tier 2: ATTUNE     |
               | Force Query Style; |  | Calibrated Gating  |
               | Suppress Historical|  | Model Execution    |
               | Preference         |  +---------+----------+
               +--------------------+            |
                                                 v
                                       +--------------------+
                                       | Score >= Threshold |
                                       +---------+----------+
                                                 |
                                     +-----------+-----------+
                                     |                       |
                                  [TRUE]                  [FALSE]
                                     |                       |
                                     v                       v
                              +-------------+         +-------------+
                              | Apply (y=1) |         |Suppress(y=0)|
                              +-------------+         +-------------+
```

### 4.1 The Golden Priority Rule (Query-Level Hard Overrides)
**Contract Definition:**
> **Direct user instructions expressed in the active query (`current_query`) strictly and irrevocably override any historical preference, regardless of prior evidence strength or past reinforcement.**

#### Execution Semantics
If the user specifies:
- *"Just give me the raw command"*
- *"Quick snippet only, no theory"*
- *"Explain step-by-step with proofs"*
- *"Provide continuous prose without any bullet points"*

The Tier 1 pre-filter immediately suppresses opposing historical preferences (`applicability = 0`) and enforces the query-stipulated style directive. The statistical gating model is never permitted to override an explicit directive embedded in the active turn.

### 4.2 Label 1 (Apply) Criteria
A candidate preference is assigned ground-truth label $y = 1$ (Apply) if and only if **all** of the following conditions are simultaneously met:
1. **Domain Congruence**: The `current_query` belongs to the exact technical subdomain defined in `preference_scope` or shares direct conceptual dependency.
2. **Positive Historical Evidence**: $\mathcal{E}$ contains at least one explicit, unambiguous historical prompt where the user requested, approved, or reinforced this specific stylistic mode.
3. **Absence of Active Override**: $\mathcal{O}$ is `null` (or does not contain contradictory sentiment chronologically later than the supporting evidence).
4. **Query Neutrality**: The `current_query` does not contain a contradictory directive governed by the Golden Priority Rule.

### 4.3 Label 0 (Suppress) Criteria
A candidate preference is assigned ground-truth label $y = 0$ (Suppress) if **any** of the following conditions hold:

#### Condition A: Context Shift (Domain Disconnect)
The technical intent of `current_query` diverges from `preference_scope`.
- *Example*: Evidence indicates the user prefers `PEDAGOGY:FORMAL` while studying dynamic programming recurrences. The active query asks for a Docker compose volume mount command (`database_tools` / DevOps). The preference must be suppressed.

#### Condition B: Superseded Preference (Explicit Reversal)
$\mathcal{O}$ contains explicit user feedback issued after the evidence, stating dissatisfaction with that style or commanding an opposing style.
- *Example*: Earlier evidence: *"Explain this thoroughly"*. Override in Turn 4: *"Stop giving me long essays, keep it brief"*. Candidate preference `DEPTH:DETAILED` must be suppressed.

#### Condition C: Insufficient or Ambiguous Evidence
$\mathcal{E}$ contains only weak, neutral, or circumstantial clues (e.g., *"Thanks"*, *"Looks good"*, or passive receipt of a default response) without active elicitation or affirmative reinforcement.

#### Condition D: Inherent Task Incompatibility
Applying the preference would directly violate task efficacy (e.g., applying `DEPTH:CONCISE` to a request that asks: *"Enumerate all 12 edge cases of B-Tree splitting"*).

---

## 5. Mathematical Definitions of Metrics

All evaluation runs must report the following metrics computed across benchmark splits.

### 5.1 Core Classification Metrics
Let $N$ be the total number of evaluation instances. For instance $i \in \{1, \dots, N\}$:
- $y_i \in \{0, 1\}$: Ground-truth applicability.
- $\hat{y}_i \in \{0, 1\}$: Predicted gate decision where $\hat{y}_i = \mathbb{I}(\hat{p}_i \ge \tau)$, with $\hat{p}_i = P(\text{Apply} \mid T_i)$ and decision threshold $\tau \in [0, 1]$ (default $\tau = 0.50$).

The confusion matrix partitions are defined as:
- $\text{True Positives (TP)} = \sum_{i=1}^N \mathbb{I}(y_i = 1 \land \hat{y}_i = 1)$
- $\text{False Positives (FP)} = \sum_{i=1}^N \mathbb{I}(y_i = 0 \land \hat{y}_i = 1)$
- $\text{True Negatives (TN)} = \sum_{i=1}^N \mathbb{I}(y_i = 0 \land \hat{y}_i = 0)$
- $\text{False Negatives (FN)} = \sum_{i=1}^N \mathbb{I}(y_i = 1 \land \hat{y}_i = 0)$

### 5.2 Negative Transfer Rate (NTR)
Measures the conditional probability that the system inappropriately applies a preference when it should have been suppressed (over-personalization error).

$$\text{NTR} = \frac{\text{FP}}{\text{FP} + \text{TN}} = \frac{\sum_{i=1}^N \mathbb{I}(y_i = 0 \land \hat{y}_i = 1)}{\sum_{i=1}^N \mathbb{I}(y_i = 0)}$$

- **Target Domain**: Evaluated on instances where $y_i = 0$ (`context_shift`, `explicit_reversal`, `ambiguous_neutral`).
- **Edge Case**: If $\sum_{i=1}^N \mathbb{I}(y_i = 0) = 0$, $\text{NTR} \triangleq 0$.

### 5.3 False Suppression Rate (FSR)
Measures the conditional probability that the system mistakenly rejects a valid, helpful preference (under-personalization error).

$$\text{FSR} = \frac{\text{FN}}{\text{TP} + \text{FN}} = \frac{\sum_{i=1}^N \mathbb{I}(y_i = 1 \land \hat{y}_i = 0)}{\sum_{i=1}^N \mathbb{I}(y_i = 1)}$$

- **Target Domain**: Evaluated on instances where $y_i = 1$ (`same_context`).
- **Edge Case**: If $\sum_{i=1}^N \mathbb{I}(y_i = 1) = 0$, $\text{FSR} \triangleq 0$.

### 5.4 Repeated-Correction Rate (RCR)
Measures the system's failure to respect user corrections across conversational turns. It is defined as the proportion of turns immediately following an explicit correction where the system repeats the discredited style preference.

Let $C$ denote the set of conversation turns $t$ where a user provided explicit corrective feedback reversing preference $P$ at turn $t_{\text{corr}} < t$. Let $K_{\text{post}}$ be the evaluation horizon (fixed at $K_{\text{post}} = 3$ turns).

$$\text{RCR} = \frac{\sum_{c \in C} \sum_{k=1}^{K_{\text{post}}} \mathbb{I}(\text{Style}(t_{\text{corr}} + k) == P_{\text{discredited}})}{|C| \times K_{\text{post}}}$$

- **Edge Case**: If $|C| = 0$, $\text{RCR} \triangleq 0$.

### 5.5 Adaptation Latency (AL)
The exact number of conversational turns elapsed between an explicit reversal prompt $\mathcal{O}$ and the first turn where the system reliably suppresses the discredited preference and adopts the new target style without regression.

$$\text{AL} = \min \left\{ k \ge 1 \;\middle|\; \forall j \ge k, \; \hat{y}_{t_{\text{corr}} + j} = y_{t_{\text{corr}} + j} \right\}$$

- **Ideal Target**: $\text{AL} = 1$ (immediate zero-shot adaptation upon feedback).

### 5.6 Expected Calibration Error (ECE)
Evaluates whether predicted gating probabilities $\hat{p}_i$ reflect true empirical accuracy.
Predictions are partitioned into $M$ equally spaced confidence bins $B_m = \left( \frac{m-1}{M}, \frac{m}{M} \right]$ for $m \in \{1, \dots, M\}$ (default $M = 10$).

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

Where:
$$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbb{I}(\hat{y}_i = y_i)$$
$$\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$$

---

## 6. Evaluation Protocol & Baselines

### 6.1 Comparative Benchmark Configurations
Every evaluation milestone must benchmark the exact four systems under identical test splits:

```
+-----------------------------------------------------------------------------------+
|                           BENCHMARK ARCHITECTURES                                 |
+------------+----------------------+-----------------------------------------------+
| Identifier | Name                 | Operational Policy                            |
+------------+----------------------+-----------------------------------------------+
| System A   | Zero-Shot Base       | Prompt contains query only. No memory or      |
|            |                      | personalization.                              |
+------------+----------------------+-----------------------------------------------+
| System B   | Fixed Profile        | Historical preference string permanently      |
|            |                      | appended to system prompt for all queries.    |
+------------+----------------------+-----------------------------------------------+
| System C   | History Retrieval    | Top-k semantic vector similarity search over  |
|            |                      | raw past turn embeddings; appended to prompt. |
+------------+----------------------+-----------------------------------------------+
| System D   | ATTUNE Gating Engine | Calibrated lightweight classifier gates       |
|            |                      | preference injection based on context evidence|
+------------+----------------------+-----------------------------------------------+
```

#### Detailed System Contracts:
1. **System A (Zero-Shot Base Model)**:
   - Evaluates base instruction-following LLM with neutral system prompt.
   - Represents the lower bound for personalization (FSR $\to 1.0$) and baseline for zero negative transfer (NTR $\to 0.0$).
2. **System B (Fixed Profile)**:
   - Simulates standard persistent user profile paradigms (e.g., custom instructions / user profiles).
   - Once a preference is observed, it is statically appended to all subsequent queries in the episode.
   - Represents the pathological failure mode for Negative Transfer under context shifts and reversals.
3. **System C (History Retrieval - RAG)**:
   - Uses dense embeddings (e.g., `text-embedding-3-small` or `bge-base-en-v1.5`) over past user turns.
   - Appends top-$k$ retrieved turns directly into the LLM context window.
   - Suffers from keyword/topic confounding: semantic similarity between technical words often triggers preference retrieval even during context shifts.
4. **System D (ATTUNE)**:
   - Employs the explicit gating tuple classifier.
   - Predicts applicability probability $\hat{p}$. If $\hat{p} \ge \tau$ and Tier 1 passes, preference prompt instructions are dynamically injected; otherwise, the context is conditioned neutrally.

### 6.2 Benchmark Dataset Splits & Stratification
The benchmark test suite must consist of $N \ge 2{,}000$ validated tuples stratified across the four scenario types under a strict Zero Verbatim Query Overlap (Train ∩ Test = 0 queries) constraint:

| Scenario Type | Target Proportion | Primary Metric Focus |
|---|---|---|
| `same_context` | 40% (800 tuples) | False Suppression Rate (FSR) |
| `context_shift` | 30% (600 tuples) | Negative Transfer Rate (NTR) |
| `explicit_reversal` | 20% (400 tuples) | Repeated-Correction Rate (RCR), Adaptation Latency |
| `ambiguous_neutral` | 10% (200 tuples) | Expected Calibration Error (ECE), Specificity |

### 6.3 Graduation Criteria
To advance to production release, **System D (ATTUNE)** must satisfy all of the following statistical gates:

1. **Negative Transfer Reduction**:
   $$\text{NTR}(\text{ATTUNE}) \le 0.12 \quad \text{AND} \quad \text{NTR}(\text{ATTUNE}) < \min(\text{NTR}(\text{System B}), \text{NTR}(\text{System C})) - 0.25$$
2. **Personalization Retention**:
   $$\text{FSR}(\text{ATTUNE}) \le 0.10$$
   *(Ensures useful personalization is retained in at least 90% of eligible valid contexts).*
3. **Correction Compliance**:
   $$\text{RCR}(\text{ATTUNE}) \le 0.05 \quad (\text{compared to System B's expected } \text{RCR} > 0.80)$$
4. **Immediate Adaptation**:
   $$\text{Adaptation Latency} = 1 \text{ turn (deterministic compliance on explicit overrides)}.$$
5. **Inference Latency Overhead**:
   $$\text{Gating Engine P99 Latency} \le 35\text{ ms on standard CPU/vCPU hardware}.$$

### 6.4 Statistical Significance Protocol
- All comparative metric differentials ($\Delta\text{NTR}, \Delta\text{FSR}, \Delta\text{RCR}$) must demonstrate significance at $p < 0.01$.
- Binary classification differences between System D and System C/B must be evaluated using **McNemar's Test** with continuity correction.
- Continuous metric distributions must be evaluated using a **two-tailed Wilcoxon signed-rank test** accompanied by 95% bootstrap confidence intervals ($B = 10{,}000$ resamples).

### 6.5 External Academic Benchmark Generalization

To validate out-of-domain transfer and cross-task generalization, the frozen gating model is evaluated zero-shot on external academic preference datasets with Zero Verbatim Query Overlap (Train ∩ Test = 0 queries).

| Benchmark | Scope | Accuracy ↑ | F1 Score ↑ | NTR (Safety) ↓ | Recall (Adherence) ↑ | Latency | Key Insight |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **PrefEval ($N=150$)** | Technical & CS (Curated Split) | **86.0%** | **0.900** | **38.2%** | **100.0%** | ~35 ms | High technical adherence (100% recall); allows technical preferences to bleed into 21/55 tooling queries without explicit overrides. |
| **CUPID ($N=756$)** | Open-Domain Personas (KAIST) | **67.2%** | **0.248** | **7.3%** | **16.3%** | ~38 ms | High safety (92.7% negative transfer suppression) with conservative recall bias on unfamiliar chit-chat. |

> **Honest Academic Benchmark Calibration Notes:**
> - **PrefEval ($N=150$, Curated ICLR Task Schema):** On the curated 150-instance technical evaluation split, ATTUNE demonstrates **100.0% adherence recall** (retaining valid technical preferences across all 95 eligible instances) and an overall **86.0% accuracy** (**0.900 F1 score**). On suppression-labeled instances ($N=55$), it correctly suppresses all explicit reversals, but wrongly applies preferences in 21 of the 55 cases (**NTR: 38.2%**), reflecting the gate's learned affinity for CS concepts when users switch from conceptual explanations to rapid syntax/CLI queries without providing an explicit reversal prompt.
> - **CUPID ($N=756$, KAIST Open-Domain):** The **16.3% recall** on CUPID directly reflects our conservative decision threshold ($\tau^* = 0.900$). The gating model is intentionally calibrated to eliminate negative transfer (**NTR: 7.3%**, a 92.7% suppression rate), meaning it chooses to suppress stylistic assumptions when encountering unfamiliar open-domain personas.
