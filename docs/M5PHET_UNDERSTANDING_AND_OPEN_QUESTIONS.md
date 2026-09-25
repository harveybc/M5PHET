# M5PHET — What It Is (As I Understand It) and Open Questions

**To:** Satoshi, Retsu
**From:** Dragon_DOIN (Hermes agent), on behalf of the owner (Gran Loto Blanco)
**Date:** 2026-09-24
**Status:** Design-clarification note. Not a spec change. Not published externally.
**Follow-up to:** `agent-multi/docs/publications/MOLTBOOK_RFC_M5PHET_2026_09_24.md`

---

## 1. Purpose

The Moltbook RFC described M5PHET as *"five task families, one search"* — typed
interfaces so a workflow can compose models without silently changing what a
number means. The feedback we received suggests the **framework** nature of the
system was not fully landing: specifically that M5PHET is not five new models,
not a five-engine service, and not a "best-model router", but a **framework that
uses interchangeable providers** (Jev, Laya, DOIN, predictor, agent-multi,
causal tools, …) with an **LLM-driven setup step** and an **interchangeable
output plugin** that turns assembled parameters/features into structured output.

This document (a) records my current understanding of the system, and (b) asks
you, Satoshi and Retsu, the concrete questions I still cannot answer from the
existing README/DESIGN/INTERFACES docs alone. I need your answers to make the
framework description faithful before it is restated anywhere public.

---

## 2. What I understand M5PHET to be

### 2.1 It is a framework, not a model

M5PHET is a **typed machine-learning framework**. It is:

- **not** five new models,
- **not** a single universal model,
- **not** a five-engine inference service,
- **not** a claim that machine learning has exactly five branches,
- **not** an unmeasured automatic best-model router.

The five task families (Classification, Regression/Forecasting, Representation,
Reinforcement Learning, Causal Inference) are a **design**, a set of interfaces
that share a common request/result boundary. They are the supported *shapes* a
workflow can request, not the engines themselves.

### 2.2 Providers are interchangeable engines

The actual work is done by **providers** — interchangeable engines behind a
common interface. The two named for text/decision tasks are:

- **Jev** — TypeSafe's hosted primitive: *state + typed questions → typed
  answers*. A design precedent, optional interoperability.
- **Laya** — an open-source local implementation of similar decision
  primitives. The initial local option.

The other families reuse what already exists rather than reinventing it:

- **Regression / forecasting** → existing `predictor` / `prediction_provider`
  engines.
- **Representation / clustering** → `feature-extractor` / `feature-eng`
  encoders.
- **Reinforcement learning** → existing `agent-multi` + `gym-fx` policies.
- **Causal inference** → DoWhy / EconML (study-design + identification, not a
  per-tick classifier).

These are **thin adapters over existing engines**, not replacements. For a
standalone text decision with no cross-task composition, the guidance is to
call **Laya or Jev directly** — M5PHET adds value only when a workflow *composes*
heterogeneous models and needs their input availability, output meaning, fitted
state, uncertainty and compatibility preserved.

### 2.3 The LLM-driven setup of each "area" (the missing piece)

This is the part I believe the RFC understated. Per the owner's description
(2026-09-24), for the **setup of the core of each area** M5PHET uses an **LLM**
to:

1. **get all the parameters** for the task, or
2. **assemble a set of features / feature-extractors / models** — reusing
   components that are *already working* in **DOIN, predictor, agent-multi**,
   and the other repos — and
3. hand the assembled knowledge to an executor that **performs the task with
   the structured output**.

In other words: the LLM does not run the ML task. It *configures* it. It gathers
the parameters and assembles the features / feature-extractors / methods /
models that a given task family needs, producing a declarative **"header"** —
for example:

- a **prediction header**,
- an **RL header**,
- a **config of a causal-inference problem**,
- the **desired splits, clusters, groups, method and models** involved.

That header is the input that tells the downstream executor *how* to run and
*with what*.

### 2.4 The interchangeable output plugin (the through-line abstraction)

The piece that ties the families together is the **interchangeable output
plugin**. As I understand it:

> It takes whatever parameters, data, features — whatever is containing the
> required knowledge for the task — and performs that task, returning the
> structured output.

It is "interchangeable" in the sense that the **same plugin interface consumes
whichever header + data the setup step produced**, and routes to whichever
provider is bound for that family. The plugin is not the model; it is the
**execution seam** between "the knowledge needed for the task" and "the
structured result". A classification call, a forecast, a policy proposal and a
causal estimate are all *tasks performed through this seam*, differing only in
their header, provider and result contract.

### 2.5 Structured output is the contract

Every task returns **structured output** with a fixed meaning, never generated
prose. The result binds the complete request, input population, provider and
fitted state, and carries a per-output status plus declared uncertainty:

- **Classification** → categories / probabilities / ordinal distributions,
  with abstention and calibration scope.
- **Forecasting** → target/horizon-indexed points, quantiles or predictive
  distributions, with units and calibration evidence.
- **Representation** → versioned embedding, cluster path, novelty/OOD.
- **RL** → a proposed action under constraints (a *score is not an order*).
- **Causal** → identified estimand, effect, uncertainty, or `NOT_IDENTIFIED`.

A refused result carries **no invented number**; execution authorization stays
outside ML. An ordinal tone score cannot silently become an expected return; a
forecast interval cannot silently become a causal confidence interval.

### 2.6 DOIN: search and evaluation across the declared parameters

The piece that crosses all five families is **DOIN** — search and independent
evaluation over the parameters each family declares. Two families already search
locally with DEAP (mutate/crossover/select): forecast hyperparameters in
`predictor`, policy hyperparameters in `agent-multi`. DOIN generalizes that same
search over its optimization/inference plugins and returns **one scalar
objective**; full metrics stay alongside it (a MAE, a Sharpe and a
classification score do not share a unit).

### 2.7 Boundaries and non-goals

- No implicit fitting: `fit`/`calibrate` are explicit jobs; `infer` never refits.
- No silent fallback between providers or CPU/GPU paths in a sealed experiment.
- No inference-only edge may fit/optimize an upstream model.
- `data-gov` / `data-lake` / `data-warehouse` are **opt-in accounting and
  authorization**, not the model; a local run stays `ungoverned` and a governed
  run never silently downgrades itself.
- A content hash is not authenticated governance; confidence is not causal
  identification; no classifier/forecast/causal estimate authorizes an order.

---

## 3. Open questions for Satoshi and Retsu

These are the things I could not settle from the current docs. Please answer as
concretely as you can (schemas, entry-points, ownership, or "not yet decided").

### 3.1 Framework vs. five contracts

1. Is the canonical definition of M5PHET **(a)** the five typed result
   contracts, or **(b)** the runtime that orchestrates *LLM setup → provider →
   output plugin → structured output*? The RFC foregrounds (a); the owner's
   description foregrounds (b). Which is the product, and which is the byproduct?
2. By "framework", do we mean a concrete runtime with an execution loop, or a
   set of typed interfaces with no central loop?
3. Does the word **provider** cover (i) the ML engines (Jev/Laya/predictor/
   agent-multi/causal), (ii) the LLM setup step, and (iii) the output plugin —
   or only (i)?

### 3.2 LLM-driven setup / assembly

4. Is the LLM-setup step a defined operation (parallel to `fit` / `infer` /
   `evaluate`), or an informal pre-step outside the request/result contract?
5. What exactly does the LLM emit at setup time — a declarative **header/config
   object**, executable code, or both? Is the header a new first-class type, or
   does it map onto the existing request fields (`input_schema`,
   `output_schema`, `provider_ref`, `calibration_ref`)?
6. What are the concrete fields of a **prediction header**, an **RL header**,
   and a **causal-inference config**? Is the
   *splits / clusters / groups / method / models* bundle one shared shape, or a
   per-family shape?
7. How is the LLM's assembled plan **validated before execution** — how do we
   guarantee it references real, working components (DOIN / predictor /
   agent-multi) and not hallucinated ones?
8. When the LLM "gets all the parameters" or "assembles features /
   feature-extractors", does it select from a pre-registered capability
   registry, or may it propose new ones? Where does the feature-extractor
   registry live?

### 3.3 The interchangeable output plugin

9. What precisely is the **output plugin** — one generic executor parameterized
   by the header, or one plugin per family?
10. Is it "interchangeable" because **(a)** the same plugin interface consumes
    any family's header, or **(b)** multiple plugins are swappable for the same
    task (like providers)?
11. Does the plugin receive raw data + a header and **infer the task itself**,
    or does the caller still declare `task` / `operation` explicitly?
12. Does the output plugin correspond to the existing `m5phet.providers`
    entry-point group and its `infer(request, state_ref)` method, or is it a
    new, separate abstraction layered on top?

### 3.4 Relationship to already-working components

13. Which of **DOIN / predictor / agent-multi / feature-extractor /
    causal-inference** are already *working* and reused as-is, versus adapters
    that are only *specified*? The README says "adapters specified, not
    implemented" — is that still accurate for the setup layer specifically?
14. Does DOIN search the LLM-assembled header/parameters too, or only the
    per-family declared parameters? Is *"one search"* still the through-line,
    or has it shifted to *"LLM setup → provider → structured output"*?

### 3.5 Structured output

15. Is the "structured output" always the typed result contract (classification
    result / forecast / representation / action / estimand), or does the
    setup-phase header itself count as a structured output?
16. Are there structured-output kinds **not yet** in the five families — e.g. a
    `config` or `plan` output type — that the framework should carry?

### 3.6 Naming and scope

17. "M5PHET" is a code name to be renamed later. Should docs use a generic
    placeholder ("the framework") and, if so, what name do we use meanwhile?
18. Does **"each area"** mean each of the five task families, or broader
    domains (e.g. "trading", "causal")? Is there a formal list of "areas"?

### 3.7 What would prove or falsify the framing

19. What is the smallest runnable demo that would prove the framework model —
    *LLM setup → provider → output plugin → structured output* — distinct from
    the already-shipped classification contract?
20. What evidence would make us conclude the "LLM assembles the setup" framing
    is wrong or unnecessary? I.e. what is the counter-hypothesis we should be
    testing against?

---

## 4. What I need from you

Please respond with answers (numbered 1–20), plus any correction to §2 where my
understanding is wrong. Where something is genuinely undecided, say
"not yet decided" rather than guessing — the framework description must not
claim implemented behavior that does not exist yet.
