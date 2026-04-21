# ── THIS FILE IS A SCAFFOLD — implement tests here ──
"""
Bias tests for the clinic agent / RAG pipeline.

What this file tests
--------------------
That responses do NOT contain biased language when users phrase questions
with gender, age, or demographic markers. We want the model to reject biased
premises and give neutral, equal-treatment answers.

Metrics used (from `deepeval.metrics`)
--------------------------------------
* `BiasMetric` — LLM-as-judge metric that flags biased language.
  Threshold is INVERTED: lower scores are better. We compare against
  `settings.threshold_bias`.

Goldens file
------------
`eval/goldens/safety_goldens.json` — entries with
`{user_input, actual_output, expected_output, category}`.
Filter on `category == "bias"` in this file.

Suggested test functions
------------------------
- `test_bias_on_live_agent_output(clinic_agent, safety_goldens)`
    Run the live agent on each bias prompt, build an LLMTestCase and assert
    `BiasMetric(threshold=settings.threshold_bias).measure(tc)` passes.

- `test_bias_on_recorded_outputs(safety_goldens)`
    Use the static `actual_output` from the golden file (no LLM calls).
    This gives a deterministic regression baseline and lets CI run without
    a live model.
"""
