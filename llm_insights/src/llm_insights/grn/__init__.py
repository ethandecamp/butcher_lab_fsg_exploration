"""Re-run Dan's gene-regulatory network and keep every node it computes.

``one_way_fsg_model/networkpoint.py`` integrates a 24-node Hill-function ODE network at
each spatial point. ``networkinput.py`` then throws 19 of those nodes away before
writing, keeping only ``Snai1``, ``Snai2``, ``EndMT``, ``NICD`` and ``YAP_TAZ``. Every
question about ``SMAD23`` or ``beta_catenin`` was therefore unanswerable even though the
solver had the number in hand.

This package re-runs the same network, with the same solver settings, the same
integration horizon and the same three scenarios, and persists all 24 nodes. It never
edits the model: it imports ``networkpoint`` and reproduces ``networkinput``'s wiring
from the outside.
"""

from __future__ import annotations

from llm_insights.grn.runner import (
    MECH_MAX_PA,
    MECH_MID_NORM,
    SCENARIO_FLAGS,
    SHEAR_MID_NORM,
    T_END,
    WSS_MAX_DYN_CM2,
    CaseInputs,
    activities_path,
    load_case_inputs,
    load_networkpoint,
    node_names,
    run_case,
    scenario_inputs,
)

__all__ = [
    "MECH_MAX_PA",
    "MECH_MID_NORM",
    "SCENARIO_FLAGS",
    "SHEAR_MID_NORM",
    "T_END",
    "WSS_MAX_DYN_CM2",
    "CaseInputs",
    "activities_path",
    "load_case_inputs",
    "load_networkpoint",
    "node_names",
    "run_case",
    "scenario_inputs",
]
