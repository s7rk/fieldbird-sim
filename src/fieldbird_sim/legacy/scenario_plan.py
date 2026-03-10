from __future__ import annotations


def build_scenarios_to_run(cfg) -> list[dict]:
    scenarios_to_run: list[dict] = []

    # Baseline always
    scenarios_to_run.append({"name": "Baseline", "kind": "baseline", "payload": None})

    # Scenario A (shifts)
    if bool(getattr(cfg.scenarios.A, "enabled", False)):
        for shift in list(getattr(cfg.scenarios.A, "shifts", [])):
            scenarios_to_run.append(
                {"name": f"Scenario_A_shift_{int(shift)}", "kind": "A", "payload": {"shift": int(shift)}}
            )

    # Scenario B (shares x fractions)
    if bool(getattr(cfg.scenarios.B, "enabled", False)):
        shares = list(getattr(cfg.scenarios.B, "shares", []))
        fracs = list(getattr(cfg.scenarios.B, "fractions", []))
        sel = str(getattr(cfg.scenarios.B, "select", "no_flower_first"))
        seedB = int(getattr(cfg.scenarios.B, "seed", 0))
        for s in shares:
            for f in fracs:
                scenarios_to_run.append(
                    {
                        "name": f"Scenario_B_share_{float(s):.2f}_frac_{float(f):.2f}",
                        "kind": "B",
                        "payload": {"share": float(s), "fraction": float(f), "select": sel, "seed": seedB},
                    }
                )

    # Scenario C (fallow shares)
    if bool(getattr(cfg.scenarios.C, "enabled", False)):
        fallows = list(getattr(cfg.scenarios.C, "fallow_area_shares", []))
        apply_years = getattr(cfg.scenarios.C, "apply_years", "all")
        selection = str(getattr(cfg.scenarios.C, "selection", "greedy"))
        seedC = int(getattr(cfg.scenarios.C, "seed", 0))
        for fa in fallows:
            scenarios_to_run.append(
                {
                    "name": f"Scenario_C_fallow_{float(fa):.2f}",
                    "kind": "C",
                    "payload": {
                        "fallow_area_share": float(fa),
                        "apply_years": apply_years,
                        "selection": selection,
                        "seed": seedC,
                    },
                }
            )

    # Scenario D (skylark plots)
    if bool(getattr(cfg.scenarios.D, "enabled", False)):
        scenarios_to_run.append(
            {
                "name": "Scenario_D_skylark",
                "kind": "D",
                "payload": {
                    "add_skylark_plots": int(getattr(cfg.scenarios.D, "add_skylark_plots", 0)),
                    "skylark_plot_area": float(getattr(cfg.scenarios.D, "skylark_plot_area", 0.0)),
                    "target_fraction": float(getattr(cfg.scenarios.D, "target_fraction", 0.0)),
                    "min_eligible_years": int(getattr(cfg.scenarios.D, "min_eligible_years", 1)),
                    "only_without_flower": bool(getattr(cfg.scenarios.D, "only_without_flower", True)),
                },
            }
        )

    return scenarios_to_run


def apply_scenario_to_inputs(
    *,
    scen: dict,
    fields_gdf_in,
    rotation_in,
    management_in,
    years_to_sim,
    apply_A_shift_management,
    apply_B_flower_strips,
    apply_C_fallow,
    apply_D_skylark,
):
    kind = scen.get("kind")
    payload = scen.get("payload") or {}

    # default: unchanged
    fields_out = fields_gdf_in
    rot_out = rotation_in
    mgmt_out = management_in

    if kind == "A":
        shift = int(payload.get("shift", 0))
        mgmt_out = apply_A_shift_management(mgmt_out, shift=shift)

    if kind == "B":
        share = float(payload.get("share", 0.0))
        fraction = float(payload.get("fraction", 0.0))
        select = str(payload.get("select", "no_flower_first"))
        seedB = int(payload.get("seed", 0))

        fields_out = apply_B_flower_strips(
            fields_out,
            share=share,
            fraction=fraction,
            select=select,
            seed=seedB,
        )

    if kind == "C":
        fallow_area_share = float(payload.get("fallow_area_share", 0.0))
        apply_years = payload.get("apply_years", "all")
        selection = str(payload.get("selection", "greedy"))
        seedC = int(payload.get("seed", 0))

        # years selector: "all" or list
        if isinstance(apply_years, str) and apply_years.strip().lower() == "all":
            years_apply = list(years_to_sim)
        elif isinstance(apply_years, (list, tuple)):
            years_apply = [int(y) for y in apply_years]
        else:
            years_apply = list(years_to_sim)

        rot_out = apply_C_fallow(
            fields_gdf_in=fields_out,
            rotation_in=rot_out,
            fallow_area_share=fallow_area_share,
            years_to_apply=years_apply,
            selection=selection,
            seed=seedC,
        )

    if kind == "D":
        # D affects fields only
        fields_out = apply_D_skylark(
            fields_gdf_in=fields_out,
            rotation_in=rot_out,
            years_to_sim=list(years_to_sim),
            add_skylark_plots=int(payload.get("add_skylark_plots", 0)),
            skylark_plot_area=float(payload.get("skylark_plot_area", 0.0)),
            target_fraction=float(payload.get("target_fraction", 0.0)),
            min_eligible_years=int(payload.get("min_eligible_years", 1)),
            only_without_flower=bool(payload.get("only_without_flower", True)),
            seed=int(payload.get("seed", 0)),  # optional; if not in payload -> 0
        )

    return fields_out, rot_out, mgmt_out
