"""The agent layer: propose hypotheses, run them, narrow what fails.

Nothing in this package decides whether a claim is true. Generators only *write*
claims; :mod:`llm_insights.harness.runner` is the only thing that judges them. Which
backend wrote a claim is recorded in ``RunResult.meta`` on every run, so a replayed
transcript can never be passed off as a live model call.
"""

from __future__ import annotations
