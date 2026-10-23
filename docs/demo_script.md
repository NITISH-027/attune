# ATTUNE: 90-Second Recruiter & Technical Interview Presentation Script

This document provides a structured, high-impact demonstration protocol for technical interviews and recruiter showcases. It is designed to be delivered in **90 seconds** while navigating the live demonstration UI at `http://127.0.0.1:8000`.

---

## 90-Second Live Walkthrough Protocol

### Step 1: The Failure Mode (0:00 – 0:30)
> **Goal:** Hook the interviewer with the fundamental flaw of current conversational AI memory.

- **Action:** Open `http://127.0.0.1:8000`. Point to the top status badges (`Accuracy: 98.5%`, `NTR: 0.0%`).
- **Spoken Script:**
  > *"Every modern AI assistant with persistent memory suffers from a critical flaw: **Negative Transfer**, or over-personalization.*
  >
  > *If you ask an AI for 'ultra-concise bullet points' while solving a dynamic programming algorithm, and then switch contexts to debug a production Git rebase conflict or understand a complex distributed system consensus failure, systems like Custom GPTs or Vector RAG blindly propagate that terse preference.*
  >
  > *In fact, our benchmarks show that standard persistent profiles produce a **100% Negative Transfer Rate** on context shifts. In high-stakes engineering, premature brevity causes missed edge cases and catastrophic outages.*
  >
  > *I built **ATTUNE** to solve this: an ultra-fast, context-aware preference gating engine."*

---

### Step 2: The Adaptive Intelligence & Live Inspector (0:30 – 1:00)
> **Goal:** Demonstrate real-time gating decisions and visual proof of safety.

- **Action 1:** Click **Preset B: Same-Domain (DSA &rarr; AVL)**.
- **Spoken Script:**
  > *"Watch what happens when the context is actually eligible. Here, the user previously requested concise answers for algorithm learning, and now asks about AVL tree rotations.*
  >
  > *ATTUNE evaluates the serialized context tuple, predicts an applicability probability of **0.9776**, clears our calibrated threshold of $\tau^* = 0.900$, and injects the concise style directive into the model context in just **37 milliseconds**."*

- **Action 2:** Click **Preset A: Context Shift (DSA &rarr; Git CLI)**.
- **Spoken Script:**
  > *"Now watch the magic when the user shifts domains to Git CLI tooling. Traditional RAG gets fooled by technical keywords. But ATTUNE detects the domain mismatch: the probability collapses to **0.0091**, well below the threshold.*
  >
  > *Look at the Policy Inspector: the amber **SUPPRESSED - OUT OF SCOPE** badge activates, and the user receives a complete, rigorous explanation with verification steps instead of a clipped snippet."*

- **Action 3:** Click **Preset C: Explicit User Override**.
- **Spoken Script:**
  > *"And when a user says 'Actually, give me a detailed breakdown and ignore my previous preference', our **Tier 1 Golden Rule** immediately intercepts the turn with **0-turn adaptation latency** (<0.1 ms), suppressing the stale preference without repeating the error."*

---

### Step 3: The Empirical Evidence & Production Graduation (1:00 – 1:30)
> **Goal:** Close with quantitative rigor, Zero Verbatim Query Overlap (Train ∩ Test = 0 queries), and external transfer.

- **Action:** Scroll to the bottom architectural summary or point to `README.md`.
- **Spoken Script:**
  > *"Under the hood, ATTUNE is a fine-tuned `all-MiniLM-L6-v2` encoder with temperature-scaled calibration ($T^* = 0.8453$, ECE $= 0.0125$).*
  >
  > *On our held-out test split of 200 strictly grouped episodes with Zero Verbatim Query Overlap (Train ∩ Test = 0 queries):*
  > 1. *Negative Transfer dropped from **100% in Fixed Profiles** and **23.3% in Rule Gates** down to **9.2% in ATTUNE** (a >90% reduction).*
  > 2. *We retained **96.25% of valid personalization** ($FSR = 3.75\%$).*
  > 3. *On external zero-shot benchmarks, ATTUNE scored **86.0% accuracy on PrefEval** (retaining 100% adherence recall across valid technical preferences, with 38.2% NTR on suppression cases) and achieved **92.7% negative transfer suppression on CUPID** (7.3% NTR, 16.3% recall).*
  >
  > *It runs entirely on standard CPU with **$37.2\text{ ms}$ average latency**, making it ready for production middleware insertion."*

---

## High-Frequency Interview Q&A

### Q1: "Why not just prompt a large LLM like Gemini or GPT-4 to remember when to personalize?"
> **Answer:**
> *"There are three major issues with LLM prompt-based routing:*
> 1. *Latency & Cost: Running an 8B–70B model just to decide whether to inject a style token adds 400–1,200 ms of latency and significant token cost to every conversational turn. ATTUNE runs in **24 ms on a single CPU core** with an 88 MB footprint.*
> 2. *Non-Deterministic Drift: Generative LLMs suffer from prompt sensitivity and sycophancy—they often agree with previous user inputs even after context shifts.*
> 3. *Safety Verification: An explicit gating engine provides auditability. Our Policy Inspector shows exact probabilities $\hat{p}$ and decision margins $\Delta p = 0.9587$, allowing engineering teams to enforce hard compliance gates."*

---

### Q2: "How did you prevent data leakage during dataset creation and evaluation?"
> **Answer:**
> *"We implemented strict **Grouped Stratified Partitioning** based on `scenario_family` and `episode_id`.
>
> In many naive ML projects, template patterns or domain topics appear in both train and test sets, causing artificial over-fitting. In ATTUNE, if the `algo_dp_knapsack_memo` family is in the test set, no knapsack problems or related templates ever appear in the training set (40 train families vs. 5 test families, verified by `data/split_and_verify.py` with 0.0% overlap).
>
> Furthermore, we validated generalization on completely external datasets—**PrefEval** (150 CS technical questions) and **CUPID** (756 open-domain dialogue episodes)—neither of which were ever exposed to training."*

---

### Q3: "Why did you use temperature scaling and threshold calibration?"
> **Answer:**
> *"Modern neural networks are notoriously overconfident: uncalibrated softmax logits tend to push probabilities toward 1.0 even on ambiguous boundary cases.
>
> 1. **Temperature Scaling ($T^*$):** We optimized a single post-hoc temperature parameter $T = 0.8453$ via L-BFGS on our held-out validation set. This reduced our Expected Calibration Error (ECE) from **0.0242 down to 0.0125** (a 48.3% improvement), ensuring that predicted probabilities accurately reflect empirical likelihood.
> 2. **Optimal Threshold ($\tau^*$):** In technical assistance, the cost of a false positive (supplying inadequate information during an outage) is far higher than a false negative. By tuning $\tau^* = 0.9000$, we slash negative transfer down to $\text{NTR} = 9.2\%$ (a 90.8% reduction compared to Fixed Profiles) while preserving 96.25% of valid preferences ($FSR = 3.75\%$)."*

---

### Q4: "How does ATTUNE compare to standard Vector RAG over user history?"
> **Answer:**
> *"Vector RAG relies on semantic cosine similarity between the current query and past user messages. But **semantic similarity is fundamentally confounded by technical terminology**.
>
> For instance, if a user previously discussed 'space complexity and tree memory footprints' in an algorithm context, and later asks 'What is the B-Tree index memory overhead in PostgreSQL?', cosine similarity between the queries is high ($\sim 0.65$) because both discuss trees and memory.
>
> RAG retrieves the past turn and injects the algorithm preference into the database query. ATTUNE, by contrast, trains the encoder to look at the **preference scope and context coherence**, cleanly rejecting cross-domain transfer."*
