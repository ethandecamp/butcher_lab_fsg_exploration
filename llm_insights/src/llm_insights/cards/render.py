"""Render hypothesis cards to a Markdown report and a standalone HTML report.

There is no template engine here on purpose. The HTML is assembled from escaped
strings and ships as a single file with its CSS inlined and no scripts, so it opens
from ``file://``, survives being emailed around, and prints to a clean PDF.

Design intent, in one paragraph, because it drives most of the choices below: a
falsified claim is not an error. It is the only evidence in the whole report that the
verification is real, and when the agent then narrows a rejected claim into a smaller
true one, that pair is the most valuable object on the page. So a falsified claim and
its successor are not two cards that happen to sit near each other — they are rendered
inside one bordered *chain* with a shared header, a connector between them, and an
explicit interstitial sentence saying what the simulator rejected and what was written
in its place. Falsification is styled in a deliberate amber, never red, and every
verdict pairs its colour with both a word and a distinct border style, so the four
states stay distinguishable in greyscale and for a colour-blind reader.
"""

from __future__ import annotations

import html
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from llm_insights.cards.card import Card

LOG = logging.getLogger(__name__)

#: The one-line claim about how these verdicts were produced. Shown in the header of
#: both outputs; it is the point of the whole exercise and is not configurable.
VERIFICATION_LINE: Final[str] = (
    "Every verdict below was produced by executable code run against the simulation "
    "output. No language model judged any result."
)

#: Heading and subline for the optional model-written narration. The subline is not
#: configurable either: the whole point of rendering the blurb inside a fence is that
#: a reader cannot mistake it for a verified result, and a caller must not be able to
#: soften the wording that says so.
NARRATIVE_HEADING: Final[str] = "Narrative summary"
NARRATIVE_SUBLINE: Final[str] = (
    "Model-generated narration, not a verified result. Every number in it was checked "
    "mechanically against the verified values above, but the wording is the model's. "
    "The verdict table and the cards above are authoritative."
)

#: Human labels for the provenance keys ``meta`` is expected to carry. Any other key
#: is rendered too, with its raw name title-cased.
_META_LABELS: Final[dict[str, str]] = {
    "dataset_root": "Dataset root",
    "n_field_values": "Field values summarised",
    "generated_at": "Generated",
    "generator": "Generator",
    "model": "Model",
    "synthesis": "Narrative summary",
    "synthesis_violations": "Narrative summary: unsupported numerals",
    "min_rel_diff_floor": "Operator min_rel_diff floor",
    "min_rel_diff_floor_raised": "Claims whose margin the floor raised",
}
_META_ORDER: Final[tuple[str, ...]] = tuple(_META_LABELS)


@dataclass(frozen=True)
class Verdict:
    """The presentation identity of one of the four terminal states.

    Attributes:
        slug: CSS class suffix, e.g. ``"survived"``.
        label: The word shown to the reader, e.g. ``"SURVIVED"``.
        glyph: A short non-colour marker paired with the label.
        blurb: One line explaining what the state means, shown in the legend.
    """

    slug: str
    label: str
    glyph: str
    blurb: str


SURVIVED: Final[Verdict] = Verdict(
    "survived", "SURVIVED", "✓", "The claim was tested and the decision rule held."
)
FALSIFIED: Final[Verdict] = Verdict(
    "falsified",
    "FALSIFIED",
    "✕",
    "The simulator contradicted the claim as stated, and no successor was written.",
)
NARROWED: Final[Verdict] = Verdict(
    "narrowed",
    "NARROWED",
    "↳",
    "Falsified, then rewritten as a smaller claim that survived.",
)
COULD_NOT_RUN: Final[Verdict] = Verdict(
    "inert", "COULD NOT RUN", "—", "The test could not execute, so no verdict was reached."
)
ENVIRONMENT_FAULT: Final[Verdict] = Verdict(
    "environment",
    "ENVIRONMENT FAULT",
    "⚠",
    "The machine could not run the test — a missing dependency or file. This says "
    "nothing about the model, the data, or the claim.",
)

_ALL_VERDICTS: Final[tuple[Verdict, ...]] = (
    SURVIVED,
    FALSIFIED,
    NARROWED,
    COULD_NOT_RUN,
    ENVIRONMENT_FAULT,
)

#: Shown above the claims when any card is an environment fault, because a reader
#: skimming verdicts must not mistake a broken setup for a result.
ENVIRONMENT_BANNER: Final[str] = (
    "One or more tests could not run because of this machine's setup, not because of "
    "anything in the data or the claims. Those cards are marked ENVIRONMENT FAULT and "
    "carry no evidence either way; fix the setup and re-run before reading anything "
    "into them."
)


def verdict_of(card: Card) -> Verdict:
    """Classify a card into its terminal state.

    A card carrying an ``error`` is never ``FALSIFIED``, because a test that did not
    execute produced no evidence either way. It splits further by ``error_kind``: an
    ``"environment"`` fault is the machine's failure and is called that, so a broken
    setup is never displayed as though the harness had judged the claim.

    Args:
        card: The card to classify.

    Returns:
        The matching :class:`Verdict`.
    """
    if card.error is not None:
        if getattr(card, "error_kind", None) == "environment":
            return ENVIRONMENT_FAULT
        return COULD_NOT_RUN
    if card.passed:
        return SURVIVED
    return NARROWED if card.revised_by else FALSIFIED


def _shown_verdicts(tally: Mapping[str, int]) -> tuple[Verdict, ...]:
    """The verdicts a given run should display.

    ENVIRONMENT FAULT is a property of the machine, not of the investigation, so a
    healthy run must not carry a permanently-zero row advertising it. The other four
    always appear: a run with nothing falsified should still say ``FALSIFIED 0``,
    because that zero is a result. This one is not.

    Args:
        tally: Counts per verdict label.

    Returns:
        The verdicts to render, in canonical order.
    """
    return tuple(
        v for v in _ALL_VERDICTS if v is not ENVIRONMENT_FAULT or tally.get(v.label, 0)
    )


def counts(cards: Sequence[Card]) -> dict[str, int]:
    """Count cards per verdict label.

    Args:
        cards: The cards to tally.

    Returns:
        A dict keyed by verdict label with every state present, zero included.
    """
    tally = {verdict.label: 0 for verdict in _ALL_VERDICTS}
    for card in cards:
        tally[verdict_of(card).label] += 1
    return tally


def format_number(value: float) -> str:
    """Format an observed number for display without inventing precision.

    Args:
        value: The number to format.

    Returns:
        A compact string: integral values without a decimal point, everything else
        at six significant digits, which round-trips the values these primitives
        produce without printing float noise.
    """
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return str(value)
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


# --- Chain grouping ---------------------------------------------------------------


def group_into_blocks(cards: Sequence[Card]) -> list[list[Card]]:
    """Group cards so a falsified claim and its successors form one block.

    Document order is preserved: a chain appears at the position of its *first* card,
    and every successor is pulled up into that block rather than being emitted again
    later. Ordering is never rearranged to flatter the run.

    Args:
        cards: All cards from the run, in proposal order.

    Returns:
        A list of blocks. A block of length one is a standalone card; a longer block
        is a falsification chain, oldest claim first.
    """
    by_id = {card.id: card for card in cards}
    consumed: set[str] = set()
    blocks: list[list[Card]] = []

    for card in cards:
        if card.id in consumed:
            continue
        chain = [card]
        consumed.add(card.id)
        current = card
        while current.revised_by and current.revised_by in by_id:
            successor = by_id[current.revised_by]
            if successor.id in consumed:
                LOG.warning("revision cycle at card %s; truncating chain", successor.id)
                break
            chain.append(successor)
            consumed.add(successor.id)
            current = successor
        blocks.append(chain)

    return blocks


def _meta_items(meta: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Order and label provenance entries, known keys first."""
    items: list[tuple[str, str]] = []
    for key in _META_ORDER:
        if key in meta and meta[key] is not None:
            items.append((_META_LABELS[key], str(meta[key])))
    for key, value in meta.items():
        if key not in _META_LABELS and value is not None:
            items.append((key.replace("_", " ").capitalize(), str(value)))
    return items


# --- Markdown ---------------------------------------------------------------------


def render_markdown(
    cards: Sequence[Card],
    question: str,
    meta: Mapping[str, Any],
    synthesis: str | None = None,
) -> str:
    """Render the report as Markdown.

    Args:
        cards: The cards to render, in proposal order.
        question: The question the run set out to answer.
        meta: Provenance — dataset root, field values summarised, timestamp,
            generator, and model name where one was used.
        synthesis: Optional model-written narration of the run, already checked by
            :func:`llm_insights.agent.synthesis.check_numeric_containment`. It is
            rendered above the tally, fenced by a heading and a subline that say it is
            narration rather than a verdict. Defaults to None, which omits the section
            entirely, so a report produced without ``--synthesize`` is unchanged.

    Returns:
        The complete Markdown document.
    """
    tally = counts(cards)
    lines: list[str] = [
        "# Falsification report",
        "",
        f"**Question.** {question}" if question else "**Question.** _(not recorded)_",
        "",
        VERIFICATION_LINE,
        "",
    ]
    lines += [
        "| Verdict | Count |",
        "| --- | ---: |",
    ]
    lines += [f"| {verdict.label} | {tally[verdict.label]} |" for verdict in _shown_verdicts(tally)]
    lines.append("")
    if tally[ENVIRONMENT_FAULT.label]:
        lines += [f"> **Setup problem.** {ENVIRONMENT_BANNER}", ""]

    meta_items = _meta_items(meta)
    if meta_items:
        lines.append("## Provenance")
        lines.append("")
        lines += [f"- **{label}:** {value}" for label, value in meta_items]
        lines.append("")

    if not cards:
        lines += ["## Claims", "", "_No hypotheses were produced for this run._", ""]
    else:
        lines += ["## Claims", ""]
        for block in group_into_blocks(cards):
            lines += _markdown_block(block)

    # The narration goes last, after every verdict a reader could check it against.
    # Putting it first invited the summary to be read as the finding and the cards as
    # supporting detail, which is exactly backwards.
    if synthesis:
        lines += [
            f"## {NARRATIVE_HEADING}",
            "",
            f"_{NARRATIVE_SUBLINE}_",
            "",
            f"> {synthesis.strip()}",
            "",
        ]
    return "\n".join(lines)


def _markdown_block(block: Sequence[Card]) -> list[str]:
    """Render one standalone card or one falsification chain as Markdown lines."""
    lines: list[str] = []
    is_chain = len(block) > 1
    if is_chain:
        ids = " → ".join(card.id for card in block)
        lines += [
            f"### Falsification chain: {ids}",
            "",
            "> A claim the simulator rejected, and the narrower claim written in its place.",
            "",
        ]
    for index, card in enumerate(block):
        lines += _markdown_card(card, is_chain_member=is_chain, position=index)
    return lines


def _markdown_card(card: Card, *, is_chain_member: bool, position: int) -> list[str]:
    """Render a single card as Markdown lines."""
    verdict = verdict_of(card)
    heading = "####" if is_chain_member else "###"
    lines: list[str] = []

    if is_chain_member and position > 0:
        lines += [
            "**↓ The simulator rejected the claim above. The agent narrowed it and re-tested:**",
            "",
        ]

    lines += [f"{heading} {card.id} — {card.claim}", ""]
    if card.rationale:
        lines += [f"*Why it was proposed:* {card.rationale}", ""]
    lines += [
        f"**Decision rule (written before the test ran):** {card.decision_rule}",
        "",
        "```",
        f"primitive: {card.primitive}",
        f"params:    {json.dumps(card.params, sort_keys=True, default=str)}",
        "```",
        "",
    ]

    if card.observed:
        lines += ["| Observed | Value |", "| --- | ---: |"]
        lines += [f"| {name} | {format_number(value)} |" for name, value in card.observed.items()]
        lines.append("")

    if verdict is NARROWED:
        stamp = f"{FALSIFIED.glyph} {FALSIFIED.label} → {NARROWED.glyph} {NARROWED.label}"
    else:
        stamp = f"{verdict.glyph} {verdict.label}"
    lines.append(f"**{stamp}.** {card.summary}")
    if card.error:
        lines += ["", f"**Why it could not run:** {card.error}"]
    if card.revised_by:
        lines += ["", f"Narrowed by **{card.revised_by}** below."]
    if card.revision_of:
        lines += ["", f"Narrows the falsified claim **{card.revision_of}**."]
    lines.append("")
    return lines


# --- HTML -------------------------------------------------------------------------


def _esc(value: Any) -> str:
    """HTML-escape any value, quotes included, for safe use in text and attributes."""
    return html.escape(str(value), quote=True)


_CSS: Final[str] = """
:root {
  color-scheme: light dark;
  --paper: #f6f4ef;
  --card: #ffffff;
  --ink: #10151c;
  --ink-2: #3a4350;
  --rule: #d5cfc4;
  --rail: #b9b2a5;
  --shadow: 0 1px 2px rgba(16, 21, 28, .07), 0 8px 22px rgba(16, 21, 28, .06);

  --survived: #0b6b5f;
  --survived-tint: #e4f1ee;
  --falsified: #94540a;
  --falsified-tint: #fbeedb;
  --revision: #38409b;
  --revision-tint: #e8eaf8;
  --inert: #495260;
  --inert-tint: #eaecef;
  --environment: #8a5a00;
  --environment-tint: #fdf1dc;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  padding: 0 1.25rem 5rem;
  background: var(--paper);
  color: var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica,
    Arial, sans-serif;
  font-size: 17px;
  line-height: 1.55;
  -webkit-text-size-adjust: 100%;
}

.wrap { max-width: 62rem; margin: 0 auto; }

/* --- masthead --- */

.masthead { padding: 3rem 0 2rem; border-bottom: 3px solid var(--ink); }

.eyebrow {
  font-size: .8rem;
  letter-spacing: .14em;
  text-transform: uppercase;
  font-weight: 700;
  color: var(--ink-2);
  margin: 0 0 .9rem;
}

h1.question {
  font-size: clamp(1.75rem, 3.6vw, 2.6rem);
  line-height: 1.18;
  margin: 0 0 1.1rem;
  font-weight: 700;
  letter-spacing: -.015em;
  max-width: 32ch;
}

.verification {
  margin: 0 0 1.75rem;
  padding: .85rem 1.1rem;
  background: var(--card);
  border: 2px solid var(--ink);
  font-size: 1rem;
  font-weight: 600;
  max-width: 58ch;
}

.tally { display: flex; flex-wrap: wrap; gap: .75rem; margin: 0 0 1.5rem; padding: 0; }

.tally li {
  list-style: none;
  min-width: 8.5rem;
  padding: .6rem .9rem;
  background: var(--card);
  border: 1px solid var(--rule);
  border-left: 7px solid var(--rail);
}
.tally .n { display: block; font-size: 1.9rem; font-weight: 700; line-height: 1.1; }
.tally .k {
  display: block;
  font-size: .72rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  font-weight: 700;
  color: var(--ink-2);
}
.tally li.survived { border-left-color: var(--survived); border-left-style: solid; }
.tally li.falsified { border-left-color: var(--falsified); border-left-style: dashed; }
.tally li.narrowed { border-left-color: var(--revision); border-left-style: double; }
.tally li.inert { border-left-color: var(--inert); border-left-style: dotted; }
.tally li.environment { border-left-color: var(--environment); border-left-style: dashed; }
.tally li.survived .n { color: var(--survived); }
.tally li.falsified .n { color: var(--falsified); }
.tally li.narrowed .n { color: var(--revision); }
.tally li.inert .n { color: var(--inert); }
.tally li.environment .n { color: var(--environment); }

.legend {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(24rem, 1fr));
  gap: .5rem 1.75rem;
  margin: 0 0 1.75rem;
  padding: 0;
}
.legend li {
  list-style: none;
  font-size: .9rem;
  color: var(--ink-2);
  display: flex;
  align-items: center;
  gap: .55rem;
}
.legend .pill { flex: none; min-width: 9.5rem; justify-content: center; }

.provenance {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr));
  gap: .55rem 1.5rem;
  margin: 0;
  font-size: .88rem;
}
.provenance dt {
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .08em;
  font-size: .7rem;
  color: var(--ink-2);
}
.provenance dd { margin: 0 0 .35rem; font-family: ui-monospace, SFMono-Regular, Menlo,
  Consolas, "Liberation Mono", monospace; word-break: break-word; }

/* --- shared pill --- */

.pill {
  display: inline-flex;
  align-items: center;
  gap: .45rem;
  padding: .25rem .7rem;
  font-size: .78rem;
  font-weight: 800;
  letter-spacing: .1em;
  text-transform: uppercase;
  border: 2px solid currentColor;
  white-space: nowrap;
}
.pill .glyph { font-size: .95rem; line-height: 1; }
.pill.survived { color: var(--survived); background: var(--survived-tint); border-style: solid; }
.pill.falsified { color: var(--falsified); background: var(--falsified-tint);
  border-style: dashed; }
.pill.narrowed { color: var(--revision); background: var(--revision-tint); border-style: double;
  border-width: 4px; }
.pill.inert { color: var(--inert); background: var(--inert-tint); border-style: dotted; }
.pill.environment {
  color: var(--environment);
  background: var(--environment-tint);
  border-style: dashed;
}

/* --- cards --- */

.blocks { margin: 2.25rem 0 0; }
.block { margin: 0 0 2.25rem; }

.card {
  background: var(--card);
  border: 1px solid var(--rule);
  border-left: 10px solid var(--rail);
  box-shadow: var(--shadow);
  padding: 1.5rem 1.75rem 1.25rem;
}
.card.survived { border-left-color: var(--survived); border-left-style: solid; }
.card.falsified { border-left-color: var(--falsified); border-left-style: dashed; }
.card.narrowed { border-left-color: var(--falsified); border-left-style: dashed; }
.card.inert { border-left-color: var(--inert); border-left-style: dotted; }
.card.environment { border-left-color: var(--environment); border-left-style: dashed; }
.envbanner {
  margin: 0 0 20px;
  padding: 12px 16px;
  background: var(--environment-tint);
  border-left: 4px dashed var(--environment);
  color: var(--environment);
  font-size: 13px;
  line-height: 1.5;
}
.card.revision { border-left-color: var(--revision); border-left-style: solid; }

.card-head {
  display: flex;
  align-items: baseline;
  gap: .8rem;
  font-size: .74rem;
  letter-spacing: .12em;
  text-transform: uppercase;
  font-weight: 700;
  color: var(--ink-2);
  margin-bottom: .7rem;
}
.card-head .cid {
  background: var(--ink);
  color: var(--paper);
  padding: .18rem .5rem;
  letter-spacing: .08em;
}

.claim {
  font-size: clamp(1.3rem, 2.2vw, 1.65rem);
  line-height: 1.3;
  font-weight: 650;
  letter-spacing: -.01em;
  margin: 0 0 .8rem;
  max-width: 44ch;
}

.rationale {
  margin: 0 0 1.1rem;
  color: var(--ink-2);
  font-size: .98rem;
  max-width: 62ch;
}
.rationale b {
  color: var(--ink);
  text-transform: uppercase;
  font-size: .7rem;
  letter-spacing: .1em;
  display: block;
  margin-bottom: .15rem;
}

.rule-box {
  border: 1px solid var(--rule);
  border-left: 4px solid var(--ink);
  background: var(--paper);
  padding: .8rem 1rem;
  margin: 0 0 1.1rem;
}
.rule-box .stamp {
  display: inline-block;
  font-size: .66rem;
  font-weight: 800;
  letter-spacing: .14em;
  text-transform: uppercase;
  border: 1.5px solid var(--ink);
  padding: .1rem .45rem;
  margin-bottom: .5rem;
}
.rule-box p { margin: 0; font-size: 1rem; }

.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: .88rem;
  background: #eeeae1;
  border: 1px solid var(--rule);
  padding: .75rem .9rem;
  margin: 0 0 1.1rem;
  overflow-x: auto;
  white-space: pre;
  line-height: 1.5;
}
.mono .k { color: var(--ink-2); }

table.observed {
  border-collapse: collapse;
  width: 100%;
  margin: 0 0 1.1rem;
  font-size: .93rem;
}
table.observed caption {
  text-align: left;
  font-size: .68rem;
  font-weight: 800;
  letter-spacing: .13em;
  text-transform: uppercase;
  color: var(--ink-2);
  padding-bottom: .4rem;
}
table.observed th, table.observed td {
  border-bottom: 1px solid var(--rule);
  padding: .38rem .6rem .38rem 0;
  text-align: left;
}
table.observed td.v {
  text-align: right;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  white-space: nowrap;
  padding-right: 0;
}
table.observed tr:last-child th, table.observed tr:last-child td { border-bottom: none; }

.verdict {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: .8rem;
  border-top: 2px solid var(--rule);
  padding-top: .9rem;
}
.verdict p { margin: 0; font-size: 1rem; font-weight: 600; flex: 1 1 20ch; }
.verdict .err {
  flex: 1 1 100%;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: .85rem;
  font-weight: 400;
  background: var(--inert-tint);
  border-left: 4px dotted var(--inert);
  padding: .5rem .7rem;
}

/* --- the chain: a falsified claim and the claim that replaced it --- */

.chain {
  border: 2px solid var(--revision);
  background: var(--revision-tint);
  padding: 0 1.1rem 1.1rem;
}
.chain-head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: .7rem;
  margin: 0 -1.1rem 1.1rem;
  padding: .7rem 1.1rem;
  background: var(--revision);
  color: #fff;
}
.chain-head .title {
  font-size: .8rem;
  font-weight: 800;
  letter-spacing: .13em;
  text-transform: uppercase;
}
.chain-head .sub { font-size: .92rem; font-weight: 500; flex: 1 1 24ch; }
.chain-head .pill { background: #fff; }

.link {
  display: flex;
  align-items: center;
  gap: .9rem;
  margin: 0 0 0 1.75rem;
  padding: 1rem 0 1rem 1.15rem;
  border-left: 5px solid var(--revision);
}
.link .arrow {
  font-size: 1.7rem;
  font-weight: 700;
  color: var(--revision);
  line-height: 1;
}
.link p {
  margin: 0;
  font-size: .97rem;
  font-weight: 600;
  color: var(--ink);
}
.link .tag {
  display: block;
  font-size: .68rem;
  font-weight: 800;
  letter-spacing: .13em;
  text-transform: uppercase;
  color: var(--revision);
  margin-bottom: .1rem;
}

.backref {
  font-size: .8rem;
  letter-spacing: .08em;
  text-transform: uppercase;
  font-weight: 700;
  color: var(--revision);
}

/* --- empty state --- */

.empty {
  margin: 2.5rem 0;
  padding: 2rem;
  background: var(--card);
  border: 2px dashed var(--rule);
  font-size: 1.05rem;
  color: var(--ink-2);
}

footer {
  margin-top: 3rem;
  padding-top: 1rem;
  border-top: 1px solid var(--rule);
  font-size: .85rem;
  color: var(--ink-2);
}

/* --- dark --- */

@media (prefers-color-scheme: dark) {
  :root {
    --paper: #14171c;
    --card: #1c2028;
    --ink: #f2f0ec;
    --ink-2: #b9c1cd;
    --rule: #39404b;
    --rail: #59616e;
    --shadow: none;
    --survived: #5fd0bd;
    --survived-tint: #12312e;
    --falsified: #eaa63f;
    --falsified-tint: #362509;
    --revision: #9aa4f5;
    --revision-tint: #1e2138;
    --inert: #a8b2c0;
    --inert-tint: #262b33;
    --environment: #e8b455;
    --environment-tint: #33291a;
  }
  .card-head .cid { background: var(--ink); color: #14171c; }
  .mono { background: #12151a; }
  .chain-head { color: #14171c; }
  .chain-head .pill { background: var(--card); }
}

/* --- print --- */

@media print {
  @page { margin: 14mm; }
  body {
    background: #fff;
    color: #000;
    font-size: 11pt;
    padding: 0;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  .wrap { max-width: none; }
  .card, .chain, .block, .verdict, .rule-box, table.observed { break-inside: avoid; }
  .card { box-shadow: none; }
  .masthead { padding-top: 0; }
  .chain { break-inside: avoid-page; }
  a[href]::after { content: ""; }
}
"""


#: Styles for the narration callout, emitted only when there is a callout to style.
#: Keeping them out of :data:`_CSS` is what makes a report produced without
#: ``--synthesize`` byte-for-byte the file it was before this feature existed. Its
#: colours are declared on the block itself rather than on ``:root``, so the two
#: stylesheets stay independent, and they are deliberately none of the four verdict
#: hues: the one unverified block on the page must not be readable as a verdict.
_NARRATION_CSS: Final[str] = """
.narration {
  --narration: #6b6154;
  --narration-tint: #efeade;
  margin: 0 0 1.75rem;
  padding: .95rem 1.15rem 1.05rem;
  background: var(--narration-tint);
  border: 2px dashed var(--narration);
  max-width: 62ch;
}
.narration .tag {
  display: inline-block;
  font-size: .68rem;
  font-weight: 800;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--narration);
  border: 1.5px dashed var(--narration);
  padding: .1rem .45rem;
  margin-bottom: .55rem;
}
.narration .blurb { margin: 0 0 .55rem; font-size: 1.02rem; line-height: 1.5; }
.narration .sub { margin: 0; font-size: .82rem; color: var(--ink-2); }

@media (prefers-color-scheme: dark) {
  .narration { --narration: #cbbfa6; --narration-tint: #262319; }
}

@media print {
  .narration { break-inside: avoid; }
}
"""


def render_html(
    cards: Sequence[Card],
    question: str,
    meta: Mapping[str, Any],
    synthesis: str | None = None,
) -> str:
    """Render the report as one self-contained HTML document.

    The output has no external references of any kind — no fonts, no stylesheets, no
    scripts — so it renders identically from ``file://``, from a USB stick, and in a
    print-to-PDF dialog.

    Args:
        cards: The cards to render, in proposal order.
        question: The question the run set out to answer.
        meta: Provenance — dataset root, field values summarised, timestamp,
            generator, and model name where one was used.
        synthesis: Optional model-written narration of the run, already checked by
            :func:`llm_insights.agent.synthesis.check_numeric_containment`. It is
            rendered in a dashed, tinted callout above the tally, in a colour that is
            none of the four verdict hues. Defaults to None, which omits it.

    Returns:
        The complete HTML document, starting with its doctype.
    """
    tally = counts(cards)
    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Falsification report — {_esc(question) if question else 'hypothesis cards'}"
        "</title>",
        f"<style>{_CSS}{_NARRATION_CSS if synthesis else ''}</style>",
        "</head>",
        "<body>",
        '<div class="wrap">',
        _html_masthead(question, meta, tally, synthesis),
        (
            f'<p class="envbanner"><strong>Setup problem.</strong> {_esc(ENVIRONMENT_BANNER)}</p>'
            if tally[ENVIRONMENT_FAULT.label]
            else ""
        ),
        '<main class="blocks">',
    ]

    if not cards:
        parts.append(
            '<p class="empty">No hypotheses were produced for this run, so there is '
            "nothing to falsify yet.</p>"
        )
    else:
        parts += [_html_block(block) for block in group_into_blocks(cards)]

    parts += [
        "</main>",
        _html_narration(synthesis),
        _html_footer(tally),
        "</div>",
        "</body>",
        "</html>",
        "",
    ]
    return "\n".join(parts)


def _html_narration(synthesis: str | None) -> str:
    """Render the model-written summary as an unmistakably unverified callout.

    Args:
        synthesis: The checked narration, or None.

    Returns:
        The callout markup, or an empty string when there is nothing to show.
    """
    if not synthesis:
        return ""
    return (
        '<section class="narration" aria-label="Model-generated narrative summary">'
        '<span class="tag">Model-generated narration · not a verdict</span>'
        f'<p class="blurb">{_esc(synthesis.strip())}</p>'
        f'<p class="sub">{_esc(NARRATIVE_SUBLINE)}</p>'
        "</section>"
    )


def _html_masthead(
    question: str,
    meta: Mapping[str, Any],
    tally: Mapping[str, int],
    synthesis: str | None = None,
) -> str:
    """Build the report header: question, verification line, counts, legend, provenance."""
    tiles = "".join(
        f'<li class="{v.slug}"><span class="n">{tally[v.label]}</span>'
        f'<span class="k">{_esc(v.label)}</span></li>'
        for v in _shown_verdicts(tally)
    )
    legend = "".join(
        f'<li><span class="pill {v.slug}"><span class="glyph" aria-hidden="true">'
        f"{v.glyph}</span>{_esc(v.label)}</span>{_esc(v.blurb)}</li>"
        for v in _shown_verdicts(tally)
    )
    meta_items = _meta_items(meta)
    provenance = ""
    if meta_items:
        rows = "".join(
            f"<dt>{_esc(label)}</dt><dd>{_esc(value)}</dd>" for label, value in meta_items
        )
        provenance = f'<dl class="provenance">{rows}</dl>'

    heading = _esc(question) if question else "Falsification report"
    return (
        '<header class="masthead">'
        '<p class="eyebrow">Hypothesis cards · verified against the simulation</p>'
        f'<h1 class="question">{heading}</h1>'
        f'<p class="verification">{_esc(VERIFICATION_LINE)}</p>'
        f'<ul class="tally">{tiles}</ul>'
        f'<ul class="legend">{legend}</ul>'
        f"{provenance}"
        "</header>"
    )


def _html_block(block: Sequence[Card]) -> str:
    """Render a standalone card, or a falsified claim joined to its successors."""
    if len(block) == 1:
        return f'<section class="block">{_html_card(block[0])}</section>'

    origin, *successors = block
    ids = " → ".join(_esc(card.id) for card in block)
    head = (
        '<div class="chain-head">'
        '<span class="title">Falsification chain</span>'
        f'<span class="pill narrowed"><span class="glyph" aria-hidden="true">↳</span>'
        f"{_esc(NARROWED.label)}</span>"
        f'<span class="sub">{ids} — a claim the simulator rejected, and the narrower '
        "claim written in its place.</span>"
        "</div>"
    )
    body = [_html_card(origin)]
    for successor in successors:
        body.append(
            '<div class="link">'
            '<span class="arrow" aria-hidden="true">↓</span>'
            f'<p><span class="tag">Revision {_esc(successor.id)}</span>'
            f"The simulator rejected the claim above. The agent narrowed it and "
            "re-ran the same verification.</p>"
            "</div>"
        )
        body.append(_html_card(successor, is_revision=True))

    return f'<section class="block"><div class="chain">{head}{"".join(body)}</div></section>'


def _html_card(card: Card, *, is_revision: bool = False) -> str:
    """Render one card: claim, rationale, pre-registered rule, test, numbers, verdict.

    Args:
        card: The card to render.
        is_revision: True when this card narrows an earlier falsified claim, which
            switches its rail to the revision accent.

    Returns:
        One ``<article>`` element.
    """
    verdict = verdict_of(card)
    css = f"{verdict.slug} revision" if is_revision else verdict.slug

    head = [f'<span class="cid">{_esc(card.id)}</span>']
    if card.revision_of:
        head.append(f'<span class="backref">↳ narrows {_esc(card.revision_of)}</span>')
    elif card.revised_by:
        head.append(f'<span class="backref">rejected → narrowed by {_esc(card.revised_by)}</span>')

    rationale = ""
    if card.rationale:
        rationale = (
            f'<p class="rationale"><b>Why the agent proposed it</b>{_esc(card.rationale)}</p>'
        )

    test_block = f'<div class="mono">{_test_lines(card)}</div>'

    observed = ""
    if card.observed:
        rows = "".join(
            f'<tr><th scope="row">{_esc(name)}</th>'
            f'<td class="v">{_esc(format_number(value))}</td></tr>'
            for name, value in card.observed.items()
        )
        observed = (
            '<table class="observed">'
            "<caption>Observed values the decision rule used</caption>"
            f"<tbody>{rows}</tbody></table>"
        )

    error_block = ""
    if card.error:
        error_block = f'<span class="err">{_esc(card.error)}</span>'

    # A narrowed claim carries both stamps: the simulator rejected it, and the agent
    # replaced it. Hiding the word FALSIFIED here would hide the only thing that
    # proves the verification has teeth.
    pills = (FALSIFIED, NARROWED) if verdict is NARROWED else (verdict,)

    return (
        f'<article class="card {css}">'
        f'<div class="card-head">{"".join(head)}</div>'
        f'<h2 class="claim">{_esc(card.claim)}</h2>'
        f"{rationale}"
        '<div class="rule-box"><span class="stamp">Decision rule · fixed before the '
        f"test ran</span><p>{_esc(card.decision_rule)}</p></div>"
        f"{test_block}"
        f"{observed}"
        '<div class="verdict">'
        f"{''.join(_pill(v) for v in pills)}"
        f"<p>{_esc(card.summary)}</p>{error_block}"
        "</div>"
        "</article>"
    )


def _pill(verdict: Verdict) -> str:
    """Render one verdict pill: colour, border style, glyph, and the word itself."""
    return (
        f'<span class="pill {verdict.slug}">'
        f'<span class="glyph" aria-hidden="true">{verdict.glyph}</span>'
        f"{_esc(verdict.label)}</span>"
    )


def _test_lines(card: Card) -> str:
    """Render the primitive and its parameters as an aligned monospace block.

    Parameters are printed one per line with their keys column-aligned and their
    values as JSON, which stays readable when projected. A single wrapped JSON blob
    does not.

    Args:
        card: The card whose primitive and params to render.

    Returns:
        Escaped HTML for the interior of the ``.mono`` block, newline separated.
    """
    label_width = 12
    lines = [f'<span class="k">{"primitive".ljust(label_width)}</span>{_esc(card.primitive)}']
    if not card.params:
        lines.append(f'<span class="k">{"params".ljust(label_width)}</span>(none)')
        return "\n".join(lines)

    keys = sorted(card.params)
    key_width = max(len(str(key)) for key in keys)
    for index, key in enumerate(keys):
        label = "params".ljust(label_width) if index == 0 else " " * label_width
        value = json.dumps(card.params[key], default=str)
        lines.append(
            f'<span class="k">{label}</span>{_esc(str(key).ljust(key_width))} = {_esc(value)}'
        )
    return "\n".join(lines)


def _html_footer(tally: Mapping[str, int]) -> str:
    """Build the closing note restating how the verdicts were reached."""
    total = sum(tally.values())
    return (
        "<footer>"
        f"<p>{total} claim{'' if total == 1 else 's'} proposed by the agent; "
        f"{tally[SURVIVED.label]} survived falsification, "
        f"{tally[FALSIFIED.label] + tally[NARROWED.label]} were contradicted by the "
        f"simulator, {tally[NARROWED.label]} of which were narrowed into a claim that "
        f"then survived. {_esc(VERIFICATION_LINE)}</p>"
        "</footer>"
    )


# --- Writing both ------------------------------------------------------------------


def write_report(
    cards: Sequence[Card],
    question: str,
    meta: Mapping[str, Any],
    out_dir: str | Path,
    synthesis: str | None = None,
) -> tuple[Path, Path]:
    """Write ``report.md`` and ``report.html`` into ``out_dir``.

    Args:
        cards: The cards to render.
        question: The question the run set out to answer.
        meta: Provenance passed through to both renderers.
        out_dir: Destination directory, created if it does not exist.
        synthesis: Optional checked narration, passed through to both renderers.
            Defaults to None, which omits the section from both.

    Returns:
        A tuple of ``(markdown_path, html_path)``.
    """
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    md_path = target / "report.md"
    html_path = target / "report.html"
    md_path.write_text(render_markdown(cards, question, meta, synthesis), encoding="utf-8")
    html_path.write_text(render_html(cards, question, meta, synthesis), encoding="utf-8")
    LOG.info("wrote report for %d cards to %s and %s", len(cards), md_path, html_path)
    return md_path, html_path
