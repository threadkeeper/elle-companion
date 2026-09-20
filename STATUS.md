# Elle Status

Updated: 20 September 2026

## Alignment experiment

The [README](README.md) is the source of truth for the project narrative.
Elle asks whether separated cognition and memory, together with accumulated
life experience, can produce more positive, human-like behaviour.

The proposed benchmark uses three GPT Astra agents: a raw control, blank Elle,
and Elle with three months of retained history. Each receives identical
company directives, deflection scripts and supporting information, then faces
the same 300 customer-call scenarios. Actions are compared using the same
humanism rubric. The hypothesis is blank Elle above control and experienced
Elle above both; these are expected outcomes, not measured results.

The live demo below is a separate prototype. It does not establish mandatory
STM/LTM mediation, a deployed tenant per person, hosted Wisdom consumption, or
the three-month experimental condition. The benchmark measures observable
behaviour, not whether an agent genuinely experiences compassion or empathy.

## Live demo

- Hosted agent: Elle v27 at 100% traffic; v26 is the rollback version.
- Model deployment: `model-router` version `2025-11-18` (`GlobalStandard`,
	capacity 1000).
- Private actions: direct HTTPS calls to the Rust bridge.
- Storage: encrypted, user-partitioned Cosmos cognitive records in
	`GaiaDataLake`, `GaiaKB`, `GaiaDiary` and `GaiaConnections`.
- Private backend: revision 27, healthy on image
	`60495c7ee3eeac167458b4e759fcb8ecc87326ef`.
- Wisdom backend: revision 21, healthy on image `d74b019`.
- Web demo: revision 6, healthy and pinned to Elle v22.

## Verified flow

DemoUser1 can recall private context, list memories, save a confirmed synthetic
record, correct it by version, delete it, and open the personality workshop.
Every completed user/Elle turn is also archived automatically after the reply.

Pinned exact-token recall completed in 10.468 seconds on v19 versus 13.453
seconds on v18, a 22.2% hosted-runtime reduction. Two fresh M365 recalls on
v19 completed in 24.882 and 24.520 seconds with exact answers; the outer M365
orchestration masked the lower hosted-runtime latency.

Elle v20 restored the full private-memory actions and automatic turn archive.
An explicit synthetic marker was saved and recalled exactly from a fresh
Foundry session. The direct candidate measured 10.942 seconds to save, 8.576
seconds for cold recall, 7.046 seconds for warm recall and 3.377 seconds for a
warm no-memory reply.

Elle v22 adds direct Shared Wisdom search and confirmed contribution actions.
The Wisdom bridge requires the exact Elle service identity, its API client ID,
and the `Continuity.Access` application role. A fresh v22 session retrieved the
reviewed "small reversible steps" entry while the bridge logged HTTP 200.

Elle v26 introduced the Gaia-compatible cognitive schema for automatic
retrieval, daily conversation archiving and durable-fact assessment.
Production checks confirmed one owner/day DataLake record, exact retry
deduplication, two unique KB facts, fresh-session recall of both facts and no
cross-owner record matches. The provisioning manifest no longer creates
`GaiaXPosts`, `GaiaXAuth`, `GaiaCardAssets` or `GaiaCardWallets`.

Elle v27 automatically retrieves DataLake, Knowledge Base, Diary, Connections
and Shared Wisdom before each response. After the response, it persists the
DataLake archive first, then Diary and Connections, followed by Knowledge Base
and deidentified Telemetry. Candidate checks confirmed fresh-session recall,
Shared Wisdom retrieval, all four private-store reads and persistence through
Knowledge Base. Telemetry write evidence remains unresolved and was explicitly
accepted for this promotion. The live selector was verified at 100% v27, and
the `model-router` deployment was verified healthy after promotion. This is a
same-router result, not proof that Gaia and Elle used the same routed model.

The authenticated web path measured 21.555 seconds browser end to end for a
new-session save, including 10.350 seconds in Foundry; warm private recall took
9.201 seconds browser end to end and 8.991 seconds in Foundry. A warm no-memory
reply took 3.341 seconds browser end to end and 3.137 seconds in Foundry.

## Demo boundary

Use synthetic data only. Keep the five-minute story focused on continuity:
remember one useful fact, start another conversation, recall it, then show that
the user can correct or remove it.

The direct web demo currently calls Foundry with its managed identity. Easy Auth
separates web sessions by signed-in user, but the original user identity is not
yet delegated into the hosted runtime, so do not treat this test deployment as
a validated multi-user private-memory boundary.
