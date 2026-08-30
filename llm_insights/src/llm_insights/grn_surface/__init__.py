"""Precomputed steady-state response surface for the GRN model.

The GRN in ``one_way_fsg_model/networkpoint.py`` is a pure function of two scalars (shear
stress and tissue stress), so its full steady-state behavior can be tabulated once and then
queried by interpolation instead of re-integrating the ODE system on every call.

See TASK-001 in ``TASKS.md`` for the implementation spec.
"""
