from __future__ import annotations

from .globals import *  # noqa: F401,F403
from .globals import _mk_outdir

#  (module 8/8) KPIs & Sensitivität (Sprint A)
# =========================================================

def _breed_mask(days: int, win: tuple[int, int]) -> np.ndarray:
    """Boolean-Maske für das Brutfenster [a,b] (1-indexed Tage)."""
    a, b = int(win[0]), int(win[1])
    a = max(1, min(a, days))
    b = max(1, min(b, days))
    if a > b:
        a, b = b, a
    m = np.zeros(days, dtype=bool)
    m[a - 1:b] = True
    return m

def field_occupancy(A_f: np.ndarray, breeding_win: tuple[int, int]) -> np.ndarray:
    """Anteil Brut-Tage mit >=1 Weibchen je Feld (Rückgabe: Vektor je Feld)."""
    days, n = A_f.shape
    m = _breed_mask(days, breeding_win)
    sub = A_f[m, :] > 0
    with np.errstate(invalid='ignore'):
        return sub.mean(axis=0)

def colonization_rate(prev: dict, curr: dict, breeding_win: tuple[int, int]) -> float:
    A_prev = prev["A_f"]; A_curr = curr["A_f"]
    days, _ = A_prev.shape
    m = _breed_mask(days, breeding_win)
    occ_prev = (A_prev[m, :] > 0).any(axis=0)
    occ_curr = (A_curr[m, :] > 0).any(axis=0)
    empty_prev = ~occ_prev
    denom = int(empty_prev.sum())
    if denom <= 0:
        return 0.0  # statt NaN
    return float(((empty_prev) & (occ_curr)).sum()) / float(denom)

def mayfield_daily_from_log(repro_df: pd.DataFrame, breeding_win: tuple[int, int]) -> float:
    """Grober Mayfield-Proxy aus dem Tageslog: mittlere Tages-Überlebensrate im Brutfenster."""
    if repro_df is None or len(repro_df) == 0:
        return float("nan")
    a, b = int(breeding_win[0]), int(breeding_win[1])
    df = repro_df[(repro_df["day"] >= a) & (repro_df["day"] <= b)]
    if len(df) == 0:
        return float("nan")
    return float(np.nanmean(df["mean_nest_surv"]))

def compute_kpis_one_year(res: dict, cfg) -> dict:
    """Kompakte Kennzahlen pro Jahr (landschaftsweit)."""
    A_m = res["A_m"]
    A_f = res["A_f"]
    days, _ = A_f.shape

    # Adults bei Ankunft
    arr_idx = max(1, int(cfg.arrival_day)) - 1
    adults_arrival = int((A_m[arr_idx, :] + A_f[arr_idx, :]).sum())

    # Geburten (flügge Jungvögel) und Todesfälle rekonstruieren
    daily_fledg = res.get("daily_fledglings")
    if daily_fledg is not None:
        daily_fledg_tot = daily_fledg.sum(axis=1)
        births_total = int(daily_fledg_tot.sum())
    else:
        daily_fledg_tot = np.zeros(days, dtype=int)
        births_total = 0
    fledglings_total = births_total  # rückwärtskompatibel

    # Erwachsene: tägliche Gesamtzahlen -> Verluste
    adults_tot = (A_m + A_f).sum(axis=1)
    adult_deaths = int(
        np.maximum(adults_tot[arr_idx:-1] - adults_tot[arr_idx + 1:], 0).sum()
    )

    # Juvenile: gebundene + freie
    J_tied = res.get("J_tied_total", res.get("J_tied"))
    if J_tied is None:
        J_tied = np.zeros_like(A_f)
    J_free = res.get("J_free", np.zeros_like(A_f))
    juven_tot = (J_tied + J_free).sum(axis=1)
    juvenile_deaths = int(
        np.maximum(
            juven_tot[arr_idx:-1] + daily_fledg_tot[arr_idx:-1] - juven_tot[arr_idx + 1:],
            0,
        ).sum()
    )

    # Female-breeding-Equivalents: Summe Brut-Weibchen-Tage / Nestzyklus
    A_f_b = res.get("A_f_breeding")
    if A_f_b is None:
        # Fallback: approximativ A_f im Brutfenster
        A_f_b = A_f.copy()
    m = _breed_mask(days, cfg.breeding_window)
    female_breed_days = int(np.asarray(A_f_b[m, :], dtype=int).sum())
    female_equiv = female_breed_days / max(int(cfg.nest_cycle_days), 1)
    fledglings_per_female_equiv = (fledglings_total / female_equiv) if female_equiv > 0 else float("nan")

    # Feld-Occupancy & Mayfield-Proxy
    occ = field_occupancy(A_f, cfg.breeding_window)
    occ_mean = float(np.nanmean(occ)) if occ.size else float("nan")
    mayfield_daily = float("nan")
    if "debug_repro" in res and isinstance(res["debug_repro"], pd.DataFrame):
        mayfield_daily = mayfield_daily_from_log(res["debug_repro"], cfg.breeding_window)

    return dict(
        adults_arrival=adults_arrival,
        fledglings_total=fledglings_total,
        female_breeding_equiv=female_equiv,
        fledglings_per_female_equiv=fledglings_per_female_equiv,
        occupancy_mean=occ_mean,
        mayfield_daily=mayfield_daily,
        adult_deaths=adult_deaths,
        juvenile_deaths=juvenile_deaths,
        births_total=births_total,
    )

def compute_kpis_from_results(results_by_year: dict, cfg) -> pd.DataFrame:
    """KPI-Tabelle je Jahr (inkl. λ zum Vorjahr und Kolonisationsrate)."""
    rows = []
    years = sorted(results_by_year.keys())

    # Adults at arrival -> λ
    arr_idx = max(1, int(cfg.arrival_day)) - 1
    adults = [int((results_by_year[y]["A_m"][arr_idx, :] + results_by_year[y]["A_f"][arr_idx, :]).sum())
              for y in years]
    lambdas = [adults[i + 1] / max(adults[i], 1e-9) for i in range(len(adults) - 1)] if len(years) > 1 else []
    lam_map = {years[i + 1]: float(lambdas[i]) for i in range(len(lambdas))}

    for i, y in enumerate(years):
        k = compute_kpis_one_year(results_by_year[y], cfg)
        k["year"] = int(y)
        k["lambda_prev"] = float(lam_map.get(y, float("nan")))
        # Kolonisation (gegenüber Vorjahr)
        if i > 0:
            k["colonization_rate"] = float(
                colonization_rate(results_by_year[years[i - 1]], results_by_year[y], cfg.breeding_window)
            )
        else:
            k["colonization_rate"] = float("nan")
        rows.append(k)
    df = pd.DataFrame(rows)
    Path("analysis").mkdir(parents=True, exist_ok=True)
    df.to_csv(Path("analysis") / "kpi_yearly.csv", index=False)
    return df

# ---------------- LHS-Sampler & PRCC ----------------

def lhs_sample(n: int, vary: dict, seed: int = 123) -> pd.DataFrame:
    """Latin Hypercube über gleichmäßige Schichten in jedem Parameterintervall [lo, hi]."""
    rng = np.random.default_rng(seed)
    names = list(vary.keys())
    k = len(names)
    # n Schichten je Dimension
    U = (rng.random((n, k)) + np.arange(n)[:, None]) / n
    X = np.zeros_like(U)
    for j, name in enumerate(names):
        lo, hi = map(float, vary[name])
        perm = rng.permutation(n)
        X[:, j] = lo + (hi - lo) * U[perm, j]
    return pd.DataFrame(X, columns=names)

def _rank_transform(a: np.ndarray) -> np.ndarray:
    """Rang-Transform (average ranks)."""
    return pd.DataFrame(a).rank(method="average").to_numpy()

def compute_prcc(params_df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """PRCC (Partial Rank Correlation Coefficients) mit Normal-Approx p-Werten (ohne SciPy)."""
    X = params_df.to_numpy(float)
    y = np.asarray(y, float).reshape(-1)

    XR = _rank_transform(X)
    yR = _rank_transform(y.reshape(-1, 1)).ravel()
    n, k = XR.shape

    out = []
    for i in range(k):
        Xi = XR[:, i]
        Xo = np.delete(XR, i, axis=1)

        # Regression Xi ~ Xo
        Bx, *_ = np.linalg.lstsq(np.c_[np.ones(n), Xo], Xi, rcond=None)
        rx = Xi - np.c_[np.ones(n), Xo] @ Bx

        # Regression yR ~ Xo
        By, *_ = np.linalg.lstsq(np.c_[np.ones(n), Xo], yR, rcond=None)
        ry = yR - np.c_[np.ones(n), Xo] @ By

        r = float(np.corrcoef(rx, ry)[0, 1])
        df = max(n - k - 1, 1)
        t = r * np.sqrt(df / max(1e-12, (1 - r**2)))
        # zweiseitiger p-Wert via Normal-Approx
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
        out.append((params_df.columns[i], r, p))

    return pd.DataFrame(out, columns=["param", "prcc", "p_approx"]).sort_values(
        "prcc", key=lambda s: s.abs(), ascending=False
    )

# -------------- MC mit LHS & KPI-Logging --------------

def run_mc_with_lhs(fields_gdf, rotation, management, cfg, years,
                    n_reps: int = 200, seed: int = 123, vary: dict | None = None) -> pd.DataFrame:
    """Monte Carlo mit LHS + KPI-Outputs (λ, Produktivität, Occupancy, Kolonisation)."""
    rng = np.random.default_rng(seed)

    # Standardbereiche (angelehnt an run_monte_carlo)
    if vary is None:
        vary = {
            "fec_scale": (0.25, 0.50),
            "fledge_rate": (0.75, 0.90),
            "alpha_density": (0.001, 0.004),
            "survival_breeding": (0.990, 0.999),
            "survival_nonbreeding": (0.998, 1.000),
            "survival_juvenile": (0.980, 0.998),
            # globals:
            "CAPACITY_GAMMA": (0.20, 0.60),
            "WINDOW_STRESS_WEIGHT": (0.40, 0.70),
            "REFUGE_ASSIGN_MULT": (10.0, 25.0),
            "PULSE_PROTECT_SCALE": (3.0, 6.0),
            "REFUGE_SHARE_CAP": (0.50, 0.80),
        }

    samples = lhs_sample(n_reps, vary, seed=seed)

    # Globale Backups (wie in run_monte_carlo)
    global CAPACITY_GAMMA, WINDOW_STRESS_WEIGHT, REFUGE_ASSIGN_MULT, PULSE_PROTECT_SCALE, REFUGE_SHARE_CAP
    _globals0 = dict(
        CAPACITY_GAMMA=CAPACITY_GAMMA,
        WINDOW_STRESS_WEIGHT=WINDOW_STRESS_WEIGHT,
        REFUGE_ASSIGN_MULT=REFUGE_ASSIGN_MULT,
        PULSE_PROTECT_SCALE=PULSE_PROTECT_SCALE,
        REFUGE_SHARE_CAP=REFUGE_SHARE_CAP,
    )

    rows = []
    for rep in range(n_reps):
        cfg2 = replace(cfg)  # flache Kopie

        # sampled Parameter anwenden
        sampled = samples.iloc[rep].to_dict()
        for k, v in sampled.items():
            if hasattr(cfg2, k):
                setattr(cfg2, k, float(v))
            elif k in _globals0:
                globals()[k] = float(v)

        # Lauf
        resY = run_years_with_feedback(fields_gdf, rotation, management, cfg2,
                                       years=years, N_init_first=50)

        # Adults + λ
        arr_idx = max(1, int(cfg2.arrival_day)) - 1
        adults = [int((resY[y]["A_m"][arr_idx, :] + resY[y]["A_f"][arr_idx, :]).sum()) for y in years]
        lambdas = [adults[i + 1] / max(adults[i], 1e-9) for i in range(len(adults) - 1)] if len(years) > 1 else []
        lam_mean = float(np.mean(lambdas)) if lambdas else float("nan")

        # KPIs pro Jahr
        kpiY = [compute_kpis_one_year(resY[y], cfg2) for y in years]
        dfY = pd.DataFrame(kpiY)

        # Kolonisation (jährlich, ab Jahr 2)
        colls = []
        for i in range(1, len(years)):
            colls.append(colonization_rate(resY[years[i - 1]], resY[years[i]], cfg2.breeding_window))
        colonization_mean = float(np.mean(colls)) if colls else float("nan")

        row = {
            "rep": rep,
            "lambda_mean": lam_mean,
            "adults_start": adults[0],
            "adults_end": adults[-1],
            "fledglings_per_female_equiv_mean": float(dfY["fledglings_per_female_equiv"].mean()),
            "occupancy_mean": float(dfY["occupancy_mean"].mean()),
            "colonization_rate_mean": colonization_mean,
            "mayfield_daily_mean": float(dfY["mayfield_daily"].mean()),
        }
        row.update({k: float(sampled[k]) for k in sampled})
        rows.append(row)

        # Globals zurücksetzen
        for k, v in _globals0.items():
            globals()[k] = v

    return pd.DataFrame(rows)

def run(use_synthetic: bool = True) -> None:
    """Execute the ported legacy simulation (formerly under if __main__)."""
    USE_SYNTHETIC = bool(use_synthetic)

    outdir = _mk_outdir("outputs")

    if USE_SYNTHETIC:
        fields_gdf = make_synthetic_fields(n_fields=int(OPTS.N_FIELDS), seed=int(OPTS.SEED_FIELDS))
        rotation = make_rotation(fields_gdf, YEARS_TO_SIM, seed=int(OPTS.SEED_ROTATION))
        management = make_management(
            fields_gdf, rotation, seed=int(OPTS.SEED_MANAGEMENT),
            use_calendar=True, calendar_path=CROP_CALENDAR_CSV,
            desync_region_sd=4.0, desync_field_sd=3.0, region_bins=6
        )
        hedge_pairs = make_random_hedges(fields_gdf, p_edge=0.7, seed=int(OPTS.SEED_HEDGES))
        hedges_gdf = build_hedge_geoms_on_shared_edges(fields_gdf, hedge_pairs)
    else:
        # CSV-Zweig wie gehabt einlesen ...
        # (YEARS_TO_SIM aus OPTS verwenden oder – falls CSV mehr/andere Jahre enthält –
        #  optional auf die ersten OPTS.N_YEARS beschränken:)
        unique_years = sorted(pd.unique(rotation["year"]))
        if len(unique_years) > len(YEARS_TO_SIM):
            YEARS_TO_SIM = unique_years[:int(OPTS.N_YEARS)]

        # Build geometry from CSV fields
        fields_gdf = rect_or_centroid_layout(fields)

        # Quick checkpoint save of the current layout (handy for debugging)
        try:
            ckpt_dir = pathlib.Path("snapshots")
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            # Overwrite a "latest" file + keep a timestamped copy
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fields_gdf.to_file(ckpt_dir / "fields_latest.gpkg", layer="fields", driver="GPKG")
            fields_gdf.to_parquet(ckpt_dir / f"fields_{ts}.parquet")
            print(f"[snapshot] fields saved to {ckpt_dir}/fields_latest.gpkg and fields_{ts}.parquet")
        except Exception as e:
            print("[warn] snapshot save failed:", e)

        # Hedges (from CSV or auto-adjacency)
        hedge_pairs = (
            pd.read_csv(HEDGES_CSV)
            if os.path.exists(HEDGES_CSV)
            else pd.DataFrame(columns=["f1", "f2", "hedge_quality"])
        )
        hedges_gdf = None
        if USE_HEDGES_FROM_CSV and {"f1", "f2"}.issubset(hedge_pairs.columns) and len(hedge_pairs) > 0:
            hedges_gdf = build_hedge_geoms_on_shared_edges(fields_gdf, hedge_pairs)
        elif AUTO_HEDGES_FROM_ADJ:
            # super-simple adjacency candidate list (optional)
            pairs = []
            fids = fields_gdf["field_id"].astype(int).values
            for i, a in enumerate(fids):
                for b in fids[i + 1:]:
                    pairs.append({"f1": int(a), "f2": int(b), "hedge_quality": 0.6})
            hedges_gdf = build_hedge_geoms_on_shared_edges(fields_gdf, pd.DataFrame(pairs)) if pairs else None

    # Sanity line so you instantly notice if the landscape got overwritten
    print(f"[Landscape] fields={len(fields_gdf)}, years={len(YEARS_TO_SIM)}, "
          f"rotation_rows={len(rotation)}, synthetic={USE_SYNTHETIC}")
    print(f"[Options] N_FIELDS={OPTS.N_FIELDS} (synthetic only) | N_YEARS={OPTS.N_YEARS} | "
          f"YEAR_TO_SHOW={YEAR_TO_SHOW} | DAY_TO_SHOW={DAY_TO_SHOW}")


    # ---------- Run simulation(s) per species ----------
    for sp_name in SPECIES_LIST:
        cfg = load_species_from_csv(sp_name, base_path=".")
        sanity_check_species(cfg)
        print("SPECIES:", cfg.name,
              "| move_radius(100m)=", cfg.move_radius,
              "| breeding_window=", cfg.breeding_window)
        print("surv_breeding=", cfg.survival_breeding,
              "surv_nonbreeding=", cfg.survival_nonbreeding,
              "surv_juvenile=", cfg.survival_juvenile,
              "fledge_rate=", cfg.fledge_rate,
              "base_nest_survival_day=", cfg.base_nest_survival_day)

        sim = BirdLandscapeSim(
            fields_gdf[[
                "field_id", "x", "y", "area", "perimeter",
                "flower_strip_share", "skylark_plots", "skylark_plot_area",
                "width_m", "height_m"
            ]],
            rotation, management, cfg
        )

        sp_outdir = outdir / cfg.name
        sp_outdir.mkdir(parents=True, exist_ok=True)

        # Base run (with cross-year feedback)
        if USE_YEAR_FEEDBACK:
            res_years = run_years_with_feedback(fields_gdf, rotation, management, cfg,
                                                years=YEARS_TO_SIM, N_init_first=100)
        else:
            res_years = {}
            for y in YEARS_TO_SIM:
                dbg_days = [110, 130, 160, 200] if y == YEAR_TO_SHOW else None
                res_years[y] = sim.simulate(days=365, year=int(y), N_init=100, debug_days=dbg_days)
                # optional: export tidy time series
                ts = to_long_timeseries(res_years[y], cfg.name, y, scenario="baseline")
                ydir = sp_outdir / f"year_{y}"; ydir.mkdir(parents=True, exist_ok=True)
                ts.to_csv(ydir / "timeseries_long.csv", index=False)

        # Bundle save (arrays + inputs + per-year CSVs)
        save_run_bundle(
            outdir=sp_outdir,
            species_name=cfg.name,
            fields_gdf=fields_gdf,
            rotation=rotation,
            management=management,
            hedges_gdf=hedges_gdf,
            results_by_year=res_years,
            day_to_show=DAY_TO_SHOW
        )
        # Ensure timeseries_long.csv exists for all years (both feedback and no-feedback modes)
        try:
            for y, res in res_years.items():
                ydir = sp_outdir / f"year_{int(y)}"
                ydir.mkdir(parents=True, exist_ok=True)
                f_long = ydir / "timeseries_long.csv"
                if not f_long.exists():
                    ts = to_long_timeseries(res, cfg.name, int(y), scenario="Baseline")
                    ts.to_csv(f_long, index=False)
        except Exception as e:
            print("[warn] could not ensure timeseries_long.csv for all years:", e)

        # Quick population summary
        arr = max(1, int(cfg.arrival_day)) - 1
        N_arrivals = [(res_years[y]["A_m"][arr, :] + res_years[y]["A_f"][arr, :]).sum() for y in YEARS_TO_SIM]
        lam = [N_arrivals[i + 1] / max(N_arrivals[i], 1e-9) for i in range(len(N_arrivals) - 1)]
        print("Adults at arrival by year:", [int(x) for x in N_arrivals])
        print("Lambda (year-to-year):", np.round(lam, 3), " mean:", round(np.mean(lam), 3))

        # Heatmaps
        if DO_HEATMAPS:
            fids = fields_gdf["field_id"].values
            vmax = max(res["N_total"].max() for res in res_years.values())
            for y in YEARS_TO_SIM:
                heatmap(
                    res_years[y]["N_total"], fids,
                    f"{cfg.name} – Heatmap Gesamt – Jahr {y}",
                    vmax=vmax,
                    savepath=sp_outdir / f"plots/heatmap_year{y}.png"
                )

        # Maps
        if DO_MAPS:
            y = YEAR_TO_SHOW
            plot_map(
                fields_gdf, rotation, year=y, hedges_gdf=hedges_gdf,
                title=f"{cfg.name} – Karte Jahr {y}, Tag {DAY_TO_SHOW}: Kultur & Vögel",
                draw_flower_rings=DRAW_FLOWER_RINGS,
                draw_hedge_buffers=DRAW_HEDGE_BUFFERS,
                hedge_buffer_m=HEDGE_BUFFER_M,
                day_vals=res_years[y]["N_total"][DAY_TO_SHOW],
                savepath=sp_outdir / f"plots/map_day{DAY_TO_SHOW}_year{y}.png"
            )
            plot_map(
                fields_gdf, rotation, year=y, hedges_gdf=hedges_gdf,
                title=f"{cfg.name} – Karte Jahr {y}: Kultur & Usage (Jahressumme)",
                draw_flower_rings=DRAW_FLOWER_RINGS,
                draw_hedge_buffers=DRAW_HEDGE_BUFFERS,
                hedge_buffer_m=HEDGE_BUFFER_M,
                usage_vals=res_years[y]["usage"],
                savepath=sp_outdir / f"plots/map_usage_year{y}.png"
            )

        # ---- Scenarios across all YEARS_TO_SIM ----
        # Verwende denselben Startbestand wie Baseline, sonst sind Vergleiche verzerrt.
        baseline_init = int(N_arrivals[0])  # adults at arrival in Jahr 1 (Baseline)


        def run_scenario_across_years(fields_df, mg_df, *, N0=baseline_init):
            return run_years_with_feedback(
                fields_df, rotation, mg_df, cfg,
                years=YEARS_TO_SIM, N_init_first=int(N0),
                outdir=None  # Szenarien schreiben keine juvenile_transition-Dateien
            )


        # ----------------- Scenarios across all YEARS_TO_SIM -----------------
        scenario_runs = {}

        # Baseline (immer 4-Tuple!)
        scenario_runs["Baseline"] = (fields_gdf.copy(), management.copy(), rotation.copy(), res_years)

        # A – spätere Ernte (mehrere Shifts als einzelne Szenarien)
        if SCENARIO_A_ENABLED:
            for dh in SCENARIO_A_SHIFTS:
                fA, mA = apply_management_scenario(
                    fields_gdf.copy(),
                    management.copy(),
                    int(dh)  # harvest_shift_days positionsbasiert
                )
                scenario_runs[f"A+{int(dh)}d"] = (fA, mA, rotation.copy(), run_scenario_across_years(fA, mA))


        # B – zusätzliche Blühstreifen (über alle Shares × Fractions)
        if SCENARIO_B_ENABLED:
            for share in SCENARIO_B_SHARES:
                for frac in SCENARIO_B_FRACTIONS:
                    fB, mB = apply_management_scenario(
                        fields_df=fields_gdf.copy(),
                        management_df=management.copy(),
                        add_flower_strip=float(share),
                        flower_fraction=float(frac),
                        flower_seed=SCENARIO_B_SEED
                    )
                    code = f"B_s{int(round(100 * share))}_f{int(round(100 * frac))}"
                    scenario_runs[code] = (fB, mB, rotation.copy(), run_scenario_across_years(fB, mB))

        # C – Brache-Anteil (Rotation ändert sich!)
        if SCENARIO_C_ENABLED:
            for share in SCENARIO_C_FALLOW_AREA_SHARES:
                rC, mC, idsC, share_act = apply_fallow_scenario(
                    fields_df=fields_gdf,
                    rotation_df=rotation,
                    management_df=management,
                    share=float(share),
                    years=YEARS_TO_SIM if SCENARIO_C_APPLY_YEARS == "all" else SCENARIO_C_APPLY_YEARS,
                    mode=SCENARIO_C_SELECTION,
                    seed=SCENARIO_C_SEED
                )
                resC = run_years_with_feedback(
                    fields_gdf, rC, mC, cfg,
                    years=YEARS_TO_SIM, N_init_first=int(baseline_init),
                    outdir=None
                )

                code = f"C_{int(round(share * 100))}p"
                scenario_runs[code] = (fields_gdf.copy(), mC, rC, resC)

        # D – gezielte Lerchenfenster
        if SCENARIO_D_ENABLED:
            target_ids_D = choose_skylark_target_fields(
                fields_gdf, rotation,
                eligible_crops=ELIGIBLE_REFUGE_CROPS,
                min_years=SCENARIO_D.get("min_eligible_years", 1),
                top_fraction=SCENARIO_D.get("target_fraction", 0.20),
                only_without_flower=SCENARIO_D.get("only_without_flower", True)
            )
            fD, mD = apply_management_scenario(
                fields_df=fields_gdf.copy(),
                management_df=management.copy(),
                add_skylark_plots=SCENARIO_D.get("add_skylark_plots", 2),
                skylark_plot_area=SCENARIO_D.get("skylark_plot_area", 0.02),
                skylark_target_field_ids=target_ids_D
            )
            scenario_runs["D"] = (fD, mD, rotation.copy(), run_scenario_across_years(fD, mD))

        # Labels für Auswertung/CSV
        scenario_labels = {
            "Baseline": "Baseline",
            "D": "D gezielte Lerchenfenster"
        }
        # A-Varianten
        if SCENARIO_A_ENABLED:
            for dh in SCENARIO_A_SHIFTS:
                scenario_labels[f"A_{int(dh)}"] = f"A Ernte +{int(dh)} Tage"
        # B-Varianten
        if SCENARIO_B_ENABLED:
            for share in SCENARIO_B_SHARES:
                for frac in SCENARIO_B_FRACTIONS:
                    code = f"B_s{int(round(100 * share))}_f{int(round(100 * frac))}"
                    scenario_labels[code] = (
                        f"B +{int(round(100 * share))}% Blühstreifen, auf {int(round(100 * frac))}% der Felder"
                    )
        # C-Varianten
        if SCENARIO_C_ENABLED:
            for share in SCENARIO_C_FALLOW_AREA_SHARES:
                scenario_labels[f"C_{int(round(share * 100))}p"] = f"C Brache {int(round(share * 100))}% Fläche"

        # ---------- Per-field/overall + Deltas schreiben ----------
        per_field_rows = []
        for scen_code, payload in scenario_runs.items():
            # payload = (fields_df, management_df, rotation_df_for_this_scenario, results_by_year)
            fDF, mDF, rotDF, resY = payload
            for yy in YEARS_TO_SIM:
                per_field_rows.append(
                    summarize_one_year(
                        res=resY[yy],
                        year=int(yy),
                        fields_df=fDF,
                        rotation_df=rotDF,  # <- korrekte Rotation je Szenario
                        species_name=cfg.name,
                        scenario_name=scenario_labels.get(scen_code, scen_code),
                        breeding_window=cfg.breeding_window
                    )
                )

        per_field_all = pd.concat(per_field_rows, ignore_index=True)
        per_field_path = sp_outdir / "summary_per_field.csv"
        per_field_all.to_csv(per_field_path, index=False)

        overall = summarize_overall(per_field_all)
        overall_path = sp_outdir / "summary_overall.csv"
        overall.to_csv(overall_path, index=False)

        deltas = add_deltas_vs_baseline(per_field_all, baseline_name="Baseline")
        deltas_path = sp_outdir / "summary_delta_vs_baseline.csv"
        deltas.to_csv(deltas_path, index=False)
        print(f"[OK] Summaries written:\n  {per_field_path}\n  {overall_path}\n  {deltas_path}")

        # ---------- KPI-Summary je Szenario (Mittel über Jahre) ----------
        kpi_rows = []
        for scen_code, (fDF, mDF, rotDF, resY) in scenario_runs.items():
            # KPIs je Jahr berechnen (ohne Datei-Schreiben)
            yrs = sorted(resY.keys())
            arr_idx = max(1, int(cfg.arrival_day)) - 1
            adults_arr = [int((resY[y]["A_m"][arr_idx, :] + resY[y]["A_f"][arr_idx, :]).sum()) for y in yrs]
            lambdas = [adults_arr[i + 1] / max(adults_arr[i], 1e-9) for i in range(len(adults_arr) - 1)] if len(
                yrs) > 1 else []

            kpiY = [compute_kpis_one_year(resY[y], cfg) for y in yrs]
            kdf = pd.DataFrame(kpiY)

            # Zusatzmetrik: Ernte im Brutfenster (Anteil)
            h_overlap = harvest_overlap_share(mDF, cfg.breeding_window)

            kpi_rows.append({
                "scenario": scenario_labels.get(scen_code, scen_code),
                "lambda_mean": float(np.mean(lambdas)) if lambdas else float("nan"),
                "adults_start": adults_arr[0] if adults_arr else float("nan"),
                "adults_end": adults_arr[-1] if adults_arr else float("nan"),
                "fledglings_per_female_equiv_mean": float(kdf["fledglings_per_female_equiv"].mean()),
                "occupancy_mean": float(kdf["occupancy_mean"].mean()),
                "colonization_rate_mean": float(
                    kdf.get("colonization_rate", pd.Series([float("nan")] * len(kdf))).mean()),
                "mayfield_daily_mean": float(kdf["mayfield_daily"].mean()),
                "harvest_in_breeding_share": float(h_overlap),
                "mean_flower_strip_share": float(fDF["flower_strip_share"].fillna(0.0).mean()),
                "share_of_fields_with_flower": float((fDF["flower_strip_share"].fillna(0.0) > 0).mean()),

            })
        kpi_summary = pd.DataFrame(kpi_rows)
        kpi_summary_path = sp_outdir / "scenario_kpis_summary.csv"
        kpi_summary.to_csv(kpi_summary_path, index=False)
        print(f"[OK] Scenario KPI summary written: {kpi_summary_path}")

        # === Sprint A: KPIs auf Baseline ===
        kpi_df = compute_kpis_from_results(res_years, cfg)
        print("\n[KPI] Jahres-Kennzahlen (Baseline):\n", kpi_df)

        # === (Optional) MC + LHS + PRCC ===
        DO_SPRINT_A_MC = False  # auf True setzen, wenn du laufen lassen willst
        if DO_SPRINT_A_MC:
            mc_df = run_mc_with_lhs(fields_gdf, rotation, management, cfg, YEARS_TO_SIM,
                                    n_reps=200, seed=123, vary=None)
            mc_path = sp_outdir / "mc_lhs_kpis.csv"
            mc_df.to_csv(mc_path, index=False)
            print(f"[OK] MC+LHS geschrieben: {mc_path}")

            # PRCC auf lambda_mean
            param_pool = [
                "fec_scale", "fledge_rate", "alpha_density",
                "survival_breeding", "survival_nonbreeding", "survival_juvenile",
                "CAPACITY_GAMMA", "WINDOW_STRESS_WEIGHT", "REFUGE_ASSIGN_MULT",
                "PULSE_PROTECT_SCALE", "REFUGE_SHARE_CAP"
            ]
            param_cols = [c for c in mc_df.columns if c in param_pool]
            prcc_df = compute_prcc(mc_df[param_cols], mc_df["lambda_mean"].to_numpy())
            print("\n[PRCC] Treiber von lambda_mean:\n", prcc_df)

        # --- NEU: Szenarien auch als birdsim_daily_full.csv speichern ---
        for scen_code, (fDF, mDF, rotDF, resY) in scenario_runs.items():
            scen_dir = sp_outdir / "scenarios" / scen_code
            scen_dir.mkdir(parents=True, exist_ok=True)

            # Daily Full Table bauen
            daily_df = _build_daily_full_df(
                fields_gdf=fDF,
                rotation_df=rotDF,
                management_df=mDF,
                results_by_year=resY,
                species_name=cfg.name,
                run_id=str(outdir.name)
            )
            if daily_df is not None:
                daily_df["scenario"] = scen_code  # Szenario-Label anhängen
                daily_df.to_csv(scen_dir / "birdsim_daily_full.csv", index=False)
                print(f"[OK] Szenario-Daily gespeichert: {scen_dir / 'birdsim_daily_full.csv'}")

        # ===== Optional: Monte Carlo (toggle on if you want to run it here) =====
        DO_MONTE_CARLO = False
        if DO_MONTE_CARLO:
            mc_df, mc_path = run_monte_carlo(
                fields_gdf, rotation, management, cfg, YEARS_TO_SIM,
                n_reps=200, seed=123, vary=None, outdir=str(sp_outdir / "mc")
            )
            print(f"[OK] Monte Carlo written: {mc_path}")

    print(f"\nAll files written under: {outdir}\n")
