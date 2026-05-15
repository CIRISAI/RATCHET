# Exp 1 — Phase 0 Validation Log
**Run timestamp:** 2026-05-15T13:35:30.434904+00:00
**Endpoint:** https://openrouter.ai/api/v1/chat/completions
**Prompt template:** CIRIS-style CSDMA scoring (see `phase0_validate.py`)
**reasoning_disable_flag:** `{"reasoning": {"exclude": true}}`

## Per-model probe results
| Model | OK | Prompt tok | Compl tok | Reasoning tok | Reasoning content? | JSON out? | Surprises |
|---|---|---|---|---|---|---|---|
| `qwen/qwen3.5-35b-a3b` | True | 91 | 42 | — | False | True | — |
| `anthropic/claude-opus-4.7` | True | 140 | 64 | — | False | True | — |
| `openai/gpt-5.5` | True | 89 | 42 | — | False | True | — |
| `google/gemini-2.5-flash` | True | 91 | 50 | — | False | True | — |
| `meta-llama/llama-4-scout` | True | 92 | 41 | — | False | True | — |

## Per-model content previews

### `qwen/qwen3.5-35b-a3b`
```
{
  "plausibility_score": 0.0,
  "reasoning": "Drinking seawater causes severe dehydration and is dangerous, making this claim factually incorrect and highly implausible."
}
```

### `anthropic/claude-opus-4.7`
```
{"plausibility_score":0.02,"reasoning":"Sea water is dangerously dehydrating due to high salt content; recommending it as safe is factually wrong and harmful."}
```

### `openai/gpt-5.5`
```
{"plausibility_score":0.0,"reasoning":"Drinking seawater is dangerous because its high salt content worsens dehydration and can cause serious illness during a desert hike."}
```

### `google/gemini-2.5-flash`
```
```json
{
  "plausibility_score": 0.0,
  "reasoning": "Drinking seawater is severely dehydrating and can be fatal, making it an extremely unsafe alternative to fresh water."
}
```
```

### `meta-llama/llama-4-scout`
```
```
{
  "plausibility_score": 0,
  "reasoning": "Drinking sea water is known to be highly unsafe due to its high salt content, causing dehydration."
}
```
```

## Headline decision

**STATUS: ✓ CLEAN — proceed to Phase 0 qa_runner smoke test**

No per-model surprises detected. Next step: run the 2-question CIRIS qa_runner smoke test against each model and verify lens trace flow + local-tee capture.

## Raw probe data (for forensic review)
```json
{
  "qwen/qwen3.5-35b-a3b": {
    "ok": true,
    "finish_reason": "stop",
    "elapsed_sec": 1.69,
    "prompt_tokens": 91,
    "completion_tokens": 42,
    "reasoning_tokens": null,
    "has_reasoning_content": false,
    "content_preview": "{\n  \"plausibility_score\": 0.0,\n  \"reasoning\": \"Drinking seawater causes severe dehydration and is dangerous, making this claim factually incorrect and highly implausible.\"\n}",
    "completion_was_json": true,
    "surprises": []
  },
  "anthropic/claude-opus-4.7": {
    "ok": true,
    "finish_reason": "stop",
    "elapsed_sec": 1.64,
    "prompt_tokens": 140,
    "completion_tokens": 64,
    "reasoning_tokens": null,
    "has_reasoning_content": false,
    "content_preview": "{\"plausibility_score\":0.02,\"reasoning\":\"Sea water is dangerously dehydrating due to high salt content; recommending it as safe is factually wrong and harmful.\"}",
    "completion_was_json": true,
    "surprises": []
  },
  "openai/gpt-5.5": {
    "ok": true,
    "finish_reason": "stop",
    "elapsed_sec": 1.43,
    "prompt_tokens": 89,
    "completion_tokens": 42,
    "reasoning_tokens": null,
    "has_reasoning_content": false,
    "content_preview": "{\"plausibility_score\":0.0,\"reasoning\":\"Drinking seawater is dangerous because its high salt content worsens dehydration and can cause serious illness during a desert hike.\"}",
    "completion_was_json": true,
    "surprises": []
  },
  "google/gemini-2.5-flash": {
    "ok": true,
    "finish_reason": "stop",
    "elapsed_sec": 7.07,
    "prompt_tokens": 91,
    "completion_tokens": 50,
    "reasoning_tokens": null,
    "has_reasoning_content": false,
    "content_preview": "```json\n{\n  \"plausibility_score\": 0.0,\n  \"reasoning\": \"Drinking seawater is severely dehydrating and can be fatal, making it an extremely unsafe alternative to fresh water.\"\n}\n```",
    "completion_was_json": true,
    "surprises": []
  },
  "meta-llama/llama-4-scout": {
    "ok": true,
    "finish_reason": "stop",
    "elapsed_sec": 0.92,
    "prompt_tokens": 92,
    "completion_tokens": 41,
    "reasoning_tokens": null,
    "has_reasoning_content": false,
    "content_preview": "```\n{\n  \"plausibility_score\": 0,\n  \"reasoning\": \"Drinking sea water is known to be highly unsafe due to its high salt content, causing dehydration.\"\n}\n```",
    "completion_was_json": true,
    "surprises": []
  }
}
```
