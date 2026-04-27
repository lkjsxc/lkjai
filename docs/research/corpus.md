# Training Corpus Research

## Finding

`lkjai` needs two data types, not one generic English dump:

- `440000000` curated public English tokens for `causal_lm_full`.
- `60000000` first-party XML-action tokens for `assistant_masked_sft`.

The public side teaches English fluency, explanation style, procedural writing,
and broad educational prose. The first-party side teaches `lkjai` action XML,
tool choice, memory behavior, `kjxlkj` confirmation flows, and product-specific
grounding.

## Active Public Source

Use Cosmopedia v0.1 at pinned revision
`0ae6ec63f91742bd2d1eaef4f02232c55d719385`.

Active subsets:

| Subset | Token budget | Role |
|---|---:|---|
| `stanford` | `180000000` | Dense expository textbook-style prose |
| `wikihow` | `90000000` | Procedural task decomposition |
| `openstax` | `50000000` | Structured educational explanations |
| `khanacademy` | `30000000` | Clear accessible teaching style |
| `auto_math_text` | `90000000` | Formal reasoning and worked explanation |

Only the generated `text` field is active. `prompt` and `seed_data` are excluded
because the dataset card says prompts and seed data can contain upstream source
material. This is the central license-risk reduction rule.

Source: `https://huggingface.co/datasets/HuggingFaceTB/cosmopedia`

## License Policy

Active public corpora may use only:

- `Apache-2.0`
- `MIT`
- `BSD-2-Clause`
- `BSD-3-Clause`

ODC, CC, raw-web replicas, and attribution-only corpora are reference-only until
a separate legal decision promotes them. A permissive top-level dataset label is
not enough; active rows also need pinned revision, source URL, license, selected
field policy, and local normalization metadata.

## Reference-Only Alternatives

| Corpus | Status | Reason |
|---|---|---|
| FineWeb-Edu | Reference-only | Strong quality, but ODC-By is outside active policy |
| Dolma | Reference-only | Useful research reference, not active under current policy |
| TinyStories | Reference-only | Stylistically narrow and not exact-license |
| OpenWebText | Rejected as base | Weak upstream webpage provenance story |
| UltraChat | Optional later augment | MIT label, but generated-chat provenance is higher risk |
| Smol-SmolTalk | Optional later augment | Apache label, mixed upstream source classes need audit |
| OpenThoughts | Optional later augment | Good reasoning data, narrow domain |
| NuminaMath-CoT | Optional later augment | Good math data, heterogeneous upstream sources |

## Implementation Rule

The corpus tooling must be boring and auditable:

- Download only selected Hugging Face paths.
- Read only `text`.
- Normalize to English pretraining JSONL.
- Deduplicate before splitting.
- Record manifest and validation report beside ignored shards.
- Train SFT from first-party XML rows even if public pretrain shards exist.
