from __future__ import annotations


def parse_year_selector(sel, *, years_to_sim, year_to_show: int) -> list[int]:
    """
    Normalize year selector config values.

    Accepted forms:
      - None -> all years
      - "all"
      - "year_to_show"
      - list/tuple of year values
    """
    if sel is None:
        return list(years_to_sim)

    if isinstance(sel, str):
        s = sel.strip().lower()
        if s == "all":
            return list(years_to_sim)
        if s == "year_to_show":
            return [int(year_to_show)]
        return list(years_to_sim)

    if isinstance(sel, (list, tuple)):
        out = []
        for x in sel:
            try:
                out.append(int(x))
            except Exception:
                pass
        valid = set(int(y) for y in years_to_sim)
        out = [y for y in out if y in valid]
        return out if out else list(years_to_sim)

    return list(years_to_sim)


def get_plot_controls(cfg, *, years_to_sim, year_to_show: int) -> dict:
    """
    Returns dict with:
      maps_years, heatmaps_years, usage_map, heatmap_vmax_mode
    """
    plots = getattr(cfg, "plots", None)

    maps_sel = getattr(plots, "maps_years", "year_to_show") if plots is not None else "year_to_show"
    heat_sel = getattr(plots, "heatmaps_years", "year_to_show") if plots is not None else "year_to_show"
    usage_map = bool(getattr(plots, "usage_map", True)) if plots is not None else True

    vmax_mode = str(getattr(plots, "heatmap_vmax_mode", "global")) if plots is not None else "global"
    vmax_mode = vmax_mode.strip().lower()
    if vmax_mode not in {"global", "per_year"}:
        vmax_mode = "global"

    return dict(
        maps_years=parse_year_selector(maps_sel, years_to_sim=years_to_sim, year_to_show=year_to_show),
        heatmaps_years=parse_year_selector(heat_sel, years_to_sim=years_to_sim, year_to_show=year_to_show),
        usage_map=usage_map,
        heatmap_vmax_mode=vmax_mode,
    )
