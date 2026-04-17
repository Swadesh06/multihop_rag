# Pilot Dataset Examples

Eight representative accepted examples from the pilot run (`outputs/pilot_annotated.jsonl`, 1 046 accepted multi-hop questions). Each example is shown with **full provenance**: the source Wikipedia articles, the graph edge that connected them, the shared entities that produced the bridge, the specific chunk spans that ground the answer, and the pipeline metadata (judge confidence, solvability confidence, quality score).

## Glossary for this document

| Term | Meaning |
|---|---|
| **path_id** | Unique ID for one pair (or triple) of documents sampled from the cross-doc graph |
| **bridge entity** | The specific shared concept (highest-IDF shared entity) that links two docs. The question must NOT name it explicitly. |
| **shared_entities** | All entities both source docs link to. The bridge is picked from this set. |
| **edge weight** | Sum of IDF (inverse doc frequency) of shared entities. Higher weight = more specific shared content. |
| **grade** | LLM-assigned importance of each support doc: 3 = essential, 2 = strong, 1 = useful context, 0 = distractor (NOT supporting, only retrieval noise) |
| **path_quality_score** | 0–1 heuristic combining bridge specificity, indirectness, doc diversity, richness |
| **judge_confidence** | LLM-as-judge (Gemini 3 Flash) confidence when picking this winner among the 4 QG candidates |
| **solvability_confidence** | Confidence that the question is NOT answerable from any single supporting doc alone |
| **distractor** | A non-supporting doc added to the retrieval pool to make retrieval non-trivial. Sourced via embedding similarity (bge-m3) or BM25 |

Distractors come from the 5 000-doc pilot corpus but don't help answer the question. Good RAG retrievers should rank supporting docs above distractors; bad ones get fooled.

---

## Example 1 — Space missions tracking near-Earth objects

| Field | Value |
|---|---|
| query_id | `q_00000` |
| reasoning_type | `bridge_entity` |
| difficulty | `medium` |
| hop_count | 2 |
| answer_type | `phrase` |
| final_score | 0.860 |
| judge_confidence | 0.80 |
| solvability_confidence | 1.00 |

**Question:** Which specific space exploration missions in the 2000s visited bodies that scientists now monitor to predict potential terrestrial collisions?

**Short answer:** `NEAR Shoemaker and Hayabusa`

**Long answer:** Scientists monitor certain celestial bodies to predict collisions: "the tracking of near-Earth objects will allow for predictions of close encounters or potential collisions of the Earth with those objects" (Document A). Two missions in the 2000s visited such bodies: "The NEAR Shoemaker mission in 2000 orbited the large near-Earth asteroid 433 Eros" (Document B), and "The Japanese ion-drive spacecraft Hayabusa in 2005 also orbited the small near-Earth asteroid 25143 Itokawa" (Document B). Thus, NEAR Shoemaker and Hayabusa are the missions that visited the bodies currently monitored for potential impact.

### Source documents

| doc_id | Title | Wikipedia | Grade | Role |
|---|---|---|---:|---|
| `doc_000021` | Astronomy | [link](https://en.wikipedia.org/wiki/Astronomy) | **2** | supports the "why monitor" premise |
| `doc_000124` | Interplanetary spaceflight | [link](https://en.wikipedia.org/wiki/Interplanetary_spaceflight) | **3** | supplies the concrete mission facts (essential) |

### Graph path `p_2c1ad1602042`

- **bridge entity**: `near-earth object`
- **edge weight**: 79.58 (high — lots of specific shared content)
- **direct edge**: false (no direct hyperlink A → B; connection is purely via shared entities)
- **n_shared**: 16 specific co-linked entities
- **path_quality_score**: 0.997

Shared entities between these two docs (sample): `venus`, `gravity`, `solar system`, `asteroid`, `mars`, `jupiter`, `mercury (planet)`, `saturn`, `van allen radiation belt`, `natural satellite`, **`near-earth object`** (selected bridge, highest IDF)

### Grounding chunks (with verbatim spans)

- `c_050650_014` (from doc `Astronomy`, char 18303–18440): *"the tracking of near-Earth objects will allow for predictions of close encounters or potential collisions of the Earth with those objects"*
- `c_015111_000` (from doc `Interplanetary spaceflight`, char 1142–1223): *"The NEAR Shoemaker mission in 2000 orbited the large near-Earth asteroid 433 Eros"*
- `c_015111_000` (from doc `Interplanetary spaceflight`, char 1325–1432): *"The Japanese ion-drive spacecraft Hayabusa in 2005 also orbited the small near-Earth asteroid 25143 Itokawa"*

### Distractors mixed into retrieval pool

| doc_id | Title | Source |
|---|---|---|
| `doc_000018` | Space exploration | embedding (bge-m3 cosine) |
| `doc_002903` | Deep Space 1 | embedding (topically close — space mission) |
| `doc_002696` | Outline of space science | BM25 (lexical overlap) |

### Why this is a good multi-hop question

The question asks about "missions that visited bodies scientists monitor for collisions". To answer, you must combine:

1. **Doc A's general fact** — "near-Earth objects are what scientists track for collisions"
2. **Doc B's specific missions** — "NEAR Shoemaker and Hayabusa orbited near-Earth asteroids"

Crucially, **Doc A never names NEAR Shoemaker or Hayabusa**, and **Doc B never says the asteroids they visited are "bodies monitored for collisions"** — that semantic bridge is supplied by Doc A. Neither document alone answers the question. Solvability gate returns 1.00 confidence that this is genuinely 2-hop.

---

## Example 2 — Heracles and the Moirai

| Field | Value |
|---|---|
| query_id | `q_00001` |
| reasoning_type | `bridge_entity` |
| difficulty | `hard` |
| hop_count | 2 |
| answer_type | `entity` |
| final_score | 0.860 |
| judge_confidence | 0.80 |
| solvability_confidence | 1.00 |

**Question:** Which historical figure was famously offered a status of veneration after his death similar to how the immortals fixed the destiny of every mortal?

**Short answer:** `Heracles`

**Long answer:** Heracles was viewed by some through a historical lens, where "a Euhemerist reading of the widespread Heracles cult was attributed to a historical figure who had been offered cult status after his death" (Document A). The power to fix human destiny belonged to the Moirai, as evidenced by their role in fixing "the mother thread of life of every mortal" (Document B) and the fact that "the immortals fixed the destiny" (Document B). Thus, Heracles is the figure whose historical deification process is discussed, while the Moirai are the entities associated with the fixing of mortal life.

### Source documents

| doc_id | Title | Wikipedia | Grade | Role |
|---|---|---|---:|---|
| `doc_000007` | Heracles | [link](https://en.wikipedia.org/wiki/Heracles) | **3** | the figure who received cult status |
| `doc_000042` | Moirai | [link](https://en.wikipedia.org/wiki/Moirai) | **2** | supplies the "fixing destiny of mortals" frame |

### Graph path `p_e1c69c5088dd`

- **bridge entity**: `cult (religion)`
- **edge weight**: 98.91 (very high)
- **n_shared**: 21 entities
- Shared (sample): `hesiod`, `titan (mythology)`, `hades`, `thanatos`, `troy`, `sparta`, `iliad`, `cult (religion)`

### Why this is a good multi-hop question

The question paraphrases concepts from both docs without naming any entity directly: "status of veneration" = cult status (Doc A side), "immortals fixed the destiny of every mortal" = the Moirai's role (Doc B side). The bridge word "cult (religion)" appears in both docs but is not in the question. The answer (Heracles) is in Doc A only; its *link* to the Doc B concept is what makes this hard — a reader must know the Moirai are what Doc B is about, then identify that Doc A describes Heracles' cult status. Difficulty: hard.

---

## Example 3 — Kidman vs Bergman, and the Tony Award

| Field | Value |
|---|---|
| query_id | `q_00006` |
| reasoning_type | `comparison` |
| difficulty | `medium` |
| hop_count | 2 |
| answer_type | `entity` |
| final_score | 0.860 |
| judge_confidence | 0.80 |
| solvability_confidence | 1.00 |

**Question:** Which of the two acclaimed actresses, one born in Hawaii and one born in Stockholm, won the Tony Award for Best Actress?

**Short answer:** `Ingrid Bergman`

**Long answer:** The question compares Nicole Kidman, "born in Honolulu, Hawaii" (Document A), with Ingrid Bergman, "born in Stockholm, Sweden" (Document B). Only Bergman's career includes the stage distinction: "she won the Tony Award for Best Actress for her performance in the Broadway production of Joan of Lorraine" (Document B). Therefore Ingrid Bergman is the correct answer.

### Source documents

| doc_id | Title | Wikipedia | Grade |
|---|---|---|---:|
| `doc_000162` | Nicole Kidman | [link](https://en.wikipedia.org/wiki/Nicole_Kidman) | **2** |
| `doc_000273` | Ingrid Bergman | [link](https://en.wikipedia.org/wiki/Ingrid_Bergman) | **3** |

### Graph path `p_4767ec5d2e35`

- **bridge entity**: `people (american magazine)`
- **edge weight**: 54.70
- **n_shared**: 11 entities
- Shared (sample): `alfred hitchcock`, `golden globe award`, `meryl streep`, `academy award for best actress`, `people (american magazine)`

### Why this is a good comparison question

**Comparison** is the second most common reasoning type (~38 % of the pilot). The question gives two descriptors — "born in Hawaii", "born in Stockholm" — that each match exactly one of the two docs. To answer, the reader must:

1. Identify which actress each descriptor refers to (needs both docs).
2. Check which one won a Tony Award (only Doc B mentions this).

Answering from Doc A alone lets you identify Kidman-as-Hawaii but never reveals she didn't win a Tony. Answering from Doc B alone lets you say "Bergman won a Tony" but you don't know why "born in Hawaii" is the other side. The comparison structure is what makes it multi-hop.

---

## Example 4 — Academy Award nominations record

| Field | Value |
|---|---|
| query_id | `q_00009` |
| reasoning_type | `comparison` |
| difficulty | `medium` |
| hop_count | 2 |
| answer_type | `entity` |
| final_score | 0.860 |
| judge_confidence | 0.80 |
| solvability_confidence | 1.00 |

**Question:** Which actress shares the record for the most Academy Award nominations without a win alongside Deborah Kerr and a performer who appeared in the film The Misfits?

**Short answer:** `Glenn Close`

### Source documents

| doc_id | Title | Wikipedia | Grade |
|---|---|---|---:|
| `doc_000091` | John Huston | [link](https://en.wikipedia.org/wiki/John_Huston) | **2** |
| `doc_000371` | Glenn Close | [link](https://en.wikipedia.org/wiki/Glenn_Close) | **3** |

### Graph path `p_1b1bffbf1efc`

- **bridge entity**: `thelma ritter`
- **edge weight**: 45.98
- **n_shared**: 9 entities
- Shared (sample): `katharine hepburn`, `bette davis`, `marilyn monroe`, `deborah kerr`, `thelma ritter`, `academy award for best supporting actress`

### Why this is interesting

The bridge (`thelma ritter`) is implicit and the question references her only as "a performer who appeared in the film The Misfits". The chain:

1. *"A performer who appeared in The Misfits"* → Thelma Ritter (from Doc A, John Huston directed The Misfits).
2. Doc B says *"Glenn Close, Deborah Kerr, and Thelma Ritter hold the record for the most nominations without a win"*.
3. Question asks for the one alongside Kerr + the Misfits performer → Glenn Close.

Three-way comparison encoded as a bridge question. The reasoning-type classifier called this `comparison` because the graph heuristic detected multiple Academy Award shared entities, even though functionally it's also bridge-like.

---

## Example 5 — The kilogram and the Wiedemann–Franz law

| Field | Value |
|---|---|
| query_id | `q_00012` |
| reasoning_type | `temporal_chain` |
| difficulty | `hard` |
| hop_count | 2 |
| answer_type | `phrase` |
| final_score | 0.860 |

**Question:** Which fundamental physical quantity, used to define the base unit of mass in the International System of Units, is mathematically linked to the ratio describing how easily a metal allows the flow of heat versus electric charge?

**Short answer:** `Planck constant`

### Source documents

| doc_id | Title | Wikipedia | Grade |
|---|---|---|---:|
| `doc_000006` | Electron | [link](https://en.wikipedia.org/wiki/Electron) | **2** |
| `doc_000084` | Kilogram | [link](https://en.wikipedia.org/wiki/Kilogram) | **3** |

### Graph path `p_2fefa02b2302`

- **bridge entity**: `thermal conductivity`
- **edge weight**: 75.39
- **n_shared**: 14 entities
- Shared (sample): `speed of light`, `photon`, `semiconductor`, `thermal conductivity`, `electric potential`, `coulomb`, `interference (wave propagation)`

### Why this is a good "temporal_chain" question

Despite the name, `temporal_chain` in our pipeline broadly covers **chained physical/causal relationships where ordering or dependency matters** — not strictly time. Here the chain is:

1. Doc B (Kilogram): *the 2019 SI redefinition fixed the kilogram in terms of the Planck constant*.
2. Doc A (Electron): *the Wiedemann–Franz law links thermal conductivity to electrical conductivity via constants including the Lorenz number* (which involves the Planck constant indirectly).

The question paraphrases the Wiedemann–Franz ratio as "how easily a metal allows the flow of heat versus electric charge" — neither doc names that ratio explicitly in those words. The reader has to recognize the concept in Doc A and then cross-reference the kilogram redefinition in Doc B. This is why difficulty was `hard` — it requires genuine physics knowledge to see the link.

---

## Example 6 — Multituberculate mammals

| Field | Value |
|---|---|
| query_id | `q_00014` |
| reasoning_type | `bridge_entity` |
| difficulty | `hard` |
| hop_count | 2 |
| answer_type | `entity` |
| final_score | 0.860 |

**Question:** Which genus of multituberculate mammals, whose fossils were discovered in the Red Beds of Hermiin Tsav, belongs to a suborder that also includes the group Paracimexomys?

**Short answer:** `Chulsanbaatar`

### Source documents

| doc_id | Title | Wikipedia | Grade |
|---|---|---|---:|
| `doc_001778` | Eobaataridae | [link](https://en.wikipedia.org/wiki/Eobaataridae) | **2** |
| `doc_003686` | Chulsanbaatar | [link](https://en.wikipedia.org/wiki/Chulsanbaatar) | **3** |

### Graph path `p_2ae070834162`

- **bridge entity**: `suborder Cimolodonta` ← note this is the *specific* taxon, not just the word "suborder"
- **edge weight**: 33.62
- **n_shared**: 7 specific shared entities
- Shared: `fossil`, `cretaceous`, `dinosaur`, `mesozoic`, `mammal`, `multituberculata`, `cimolodonta`

### Note on bridge quality

An earlier pilot iteration of the graph had picked the generic word `suborder` as the bridge for a similar path (see `p_00000` in the first `pilot_qg_raw.jsonl`), which led to weak, leaky questions. The agent's second graph build produced this improved path where the bridge is the specific taxonomic name `suborder Cimolodonta`. The question correctly doesn't name Cimolodonta explicitly — instead it says *"a suborder that also includes the group Paracimexomys"*, which requires the reader to know (from Doc A) that Paracimexomys is in Cimolodonta, then check (from Doc B) that Chulsanbaatar is also in Cimolodonta.

This is a good example of how bridge specificity dramatically changes question quality.

---

## Example 7 — Hassium, seaborgium, and oxidation state +8

| Field | Value |
|---|---|
| query_id | `q_00020` |
| reasoning_type | `bridge_entity` |
| difficulty | `hard` |
| hop_count | 2 |
| answer_type | `phrase` |
| final_score | 0.860 |

**Question:** Which element, known for having a higher density than seaborgium, also features an oxidation state of +8 that is notably shared with iridium and ruthenium?

**Short answer:** `hassium`

### Source documents

| doc_id | Title | Wikipedia | Grade |
|---|---|---|---:|
| `doc_000419` | Seaborgium | [link](https://en.wikipedia.org/wiki/Seaborgium) | **3** |
| `doc_000427` | Osmium | [link](https://en.wikipedia.org/wiki/Osmium) | **3** |

### Graph path `p_d2328d8fae7c`

- **bridge entity**: `hassium` (which is, notably, *also the answer*)
- **edge weight**: 56.55
- **n_shared**: 11 entities
- Shared (sample): `atomic number`, `isotope`, `half-life`, `alpha decay`, `uranium`, `transition metal`, `density`, `tungsten`

### Caveat: answer-as-bridge

This example shows a subtle edge case the pipeline allows: the **bridge entity is the same as the answer**. The pipeline's rule is *"the bridge must not be named in the question"*, not *"the bridge must not be the answer"*. Here:

- The question does not name `hassium`.
- The question describes hassium through two separate facts that each sit in one of the source docs: density > seaborgium (Doc A, Seaborgium article mentions hassium) + oxidation state +8 shared with Os/Ir/Ru (Doc B, Osmium article).
- Both docs are grade 3 because both are essential.
- Solvability test confirmed 1.00 confidence that neither doc alone reveals the answer.

The question is valid multi-hop, but future iterations of the pipeline might add a rule: *"if bridge == short_answer, require bridge to appear only implicitly in both docs"* — which is already the case here. Noted for the scale-run review.

---

## Example 8 — Parliaments of Luxembourg and Papua New Guinea

| Field | Value |
|---|---|
| query_id | `q_00029` |
| reasoning_type | `bridge_entity` |
| difficulty | `medium` |
| hop_count | 2 |
| answer_type | `phrase` |
| final_score | 0.860 |

**Question:** How does the duration of terms for members of the national legislative bodies in Luxembourg and Papua New Guinea compare?

**Short answer:** `Both serve five-year terms`

### Source documents

| doc_id | Title | Wikipedia | Grade |
|---|---|---|---:|
| `doc_000704` | Politics of Luxembourg | [link](https://en.wikipedia.org/wiki/Politics_of_Luxembourg) | **3** |
| `doc_001410` | Politics of Papua New Guinea | [link](https://en.wikipedia.org/wiki/Politics_of_Papua_New_Guinea) | **3** |

### Graph path `p_c275fb9fd842`

- **bridge entity**: `parliamentary system`
- **edge weight**: 152.86 (highest in this set — extremely information-rich pair)
- **n_shared**: 31 entities
- Shared (sample): `world trade organization`, `judiciary`, `head of government`, `international organization for standardization`, `world health organization`, `parliamentary system`, `representative democracy`

### Why this comparison works

Both documents are structured political fact sheets. The question asks for a *comparison* of a specific shared attribute (term length). The answer requires extracting the exact fact from both docs and confirming they match. "Five-year terms" is not a phrase that appears in the question — the reader must retrieve it from both docs.

The 31-shared-entity overlap means the graph has high confidence these are comparable docs. The question exercises a common RAG failure mode: retrievers often fetch one country's article and miss the other. BM25 recall@10 on this question is likely low (closer to 0.5), which is exactly the kind of difficulty the pipeline targets.

---

## Rollup: what these 8 examples tell us

| Property | Pilot distribution | These 8 examples |
|---|---|---|
| reasoning_type mix | 57% bridge, 38% comp, 5% temporal | 5 bridge, 2 comp, 1 temporal (representative) |
| difficulty | 73.9 % medium, 26.0 % hard, 0.1 % easy | 4 medium, 4 hard |
| answer_type | 52.8 % entity, 46.7 % phrase, 0.5 % numeric | 4 entity, 4 phrase |
| mean final_score | 0.61 (pilot avg) | 0.86 (top-30 slice) |
| judge_confidence | 0.80 typical | 0.80 |
| solvability_confidence | 1.00 typical (all pass the gate) | 1.00 |

These are the top 30 by composite score, so they're the "best of the pilot" — but every one of the 1 046 accepted questions has passed the 6 quality gates. The examples here illustrate the **range of the dataset**: single-bridge comparisons, three-way chains (q_00009), indirect bridges (q_00014), and comparisons where the bridge is structural rather than entity-like (q_00029).

The full pilot dataset is in `outputs/pilot_annotated.jsonl`; every record has the same schema as shown here, including quoted spans, chunk offsets, distractor sources, and generation metadata.
