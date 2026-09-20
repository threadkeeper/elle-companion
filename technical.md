# Elle technical guide

## Research goal and implementation boundary

The [README](README.md) defines Elle's alignment hypothesis: separate raw LLM
cognition from short-term and long-term memory, mediate outside interactions
through those memory layers, and give each agent a unique retained history.
The proposed trust model combines a tenant for each person with an opt-in
Wisdom layer for anonymized lessons rather than raw private histories.

The proposed experiment compares a raw GPT Astra endpoint, a blank Elle agent
on GPT Astra, and an Elle agent on GPT Astra with three months of retained
history. All receive the same company directives, deflection scripts and
supporting information for the same 300 customer-call scenarios. Chosen actions
are scored against a common humanism rubric. The hypothesis is that blank Elle
outperforms the raw control and experienced Elle outperforms both.

The runtime below is a prototype foundation, not a completed implementation or
evaluation of that research design. Its current Luna model is distinct from the
proposed Astra experimental conditions. User-partitioned storage is not a
separate deployed tenant for every person, and hosted Elle does not currently
consume Shared Wisdom. No 300-scenario results or genuine compassion/empathy
claims are reported.

## Runtime

The demo is a Microsoft Foundry hosted agent. Its Python runtime uses the
Foundry Chat client and the Responses host. Private memory is registered as
ordinary Python functions in `foundry-agent/private_tools.py`.

`foundry-agent/turn_memory.py` retrieves the latest two DataLake days and up to
12 relevant knowledge facts before each response. After completion it queues a
background lifecycle task that appends the exact turn to the UTC-day DataLake
document, runs a strict structured knowledge assessment, and saves each durable
fact to KB. The post-response work does not delay the visible reply.

Each function sends JSON over HTTPS to the existing Rust service at
`/bridge/{tool}`. The request includes the current platform user identity and
the service resolves its SHA-256 continuity handle through the configured
binding map before selecting a private partition. Unknown handles fail closed.
No separate tool catalog or connector is involved in the agent runtime.

## Private actions

- `elle_personality`: open the workshop.
- `elle_set_personality`: save a confirmed profile version.
- `elle_cognitive_query`: retrieve DataLake, knowledge, diary or connection
	records with bounded mode, date, order, count and result options.
- `elle_save_cognitive`: save a knowledge fact or diary reflection; automatic
	KB writes use the same validated contract.
- `elle_archive_turn`: internal post-response DataLake archival.

The former `elle_context`, `elle_list_memories`, `elle_remember`, `elle_correct`
and `elle_forget` contracts are no longer advertised or registered with Elle.
Existing old-schema data is retained unchanged for compatibility and migration;
the cognitive lifecycle does not read or write it.

The Rust service validates action payloads, derives the owner partition from the
caller identity, encrypts private fields, and uses Cosmos DB for persistence.

## Cognitive Cosmos stores

The cognitive runtime uses `GaiaDataLake`, `GaiaKB`, `GaiaDiary` and
`GaiaConnections` in the database selected by `ELLE_COSMOS_DATABASE`. Every
runtime store is partitioned by `/owner_id`; the owner is never accepted in a
tool payload. DataLake and Diary keep one compare-and-swap document per UTC day,
while KB stores one canonical fact per document and requires salience from 0 to
1. Medium retrieval defaults are DataLake 2, KB 12, Diary 12 and Connections 3.

Unlike Gaia, Elle encrypts cognitive text and embeddings together. Semantic
ranking therefore runs in the trusted Rust process after a bounded owner-scoped
Cosmos read. The provisioning policy intentionally omits Cosmos vector and
full-text indexes because those services cannot index the encrypted payload.

Provision all Gaia-equivalent primary container names in the existing Elle
database with the isolated script dependencies:

```powershell
$env:ELLE_COSMOS_ENDPOINT = 'https://<account>.documents.azure.com'
$env:ELLE_COSMOS_DATABASE = '<existing-elle-database>'
uv run --with-requirements scripts/requirements.txt python scripts/cosmos_create.py --dry-run
uv run --with-requirements scripts/requirements.txt python scripts/cosmos_create.py
```

The script uses `DefaultAzureCredential`, validates existing partition,
unique-key and indexing policies, and never changes `private-memory` or
`wisdom-settings`.

## Deployment

`foundry-agent/deploy.py` packages the Python runtime and stages an explicit
Foundry agent version. It configures:

- `AZURE_AI_MODEL_DEPLOYMENT_NAME`
- `ELLE_PRIVATE_TOOLS_ENDPOINT`

Promotion remains guarded by an expected live version. The default model is
`gpt-5.6-luna`; `model-router` remains available through the environment
override.

## Demo discipline

Use fictional records. Keep the prompt and tool responses concise. Automatic
turn archives and attained-knowledge saves require no confirmation; diary and
personality changes keep their normal confirmation flow. Run the focused Python
and Rust tests before staging a candidate.
