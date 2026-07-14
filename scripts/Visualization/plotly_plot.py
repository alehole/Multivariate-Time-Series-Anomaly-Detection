from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import config as cfg

# -----------------------------
# Helpers
# -----------------------------
def find_timestamp_col(
    df: pd.DataFrame,
    preferred=("Created", "Modified", "Inserted", "Timestamp", "Time"),
) -> str | None:
    for c in preferred:
        if c in df.columns:
            return c
    return None


def time_span_from_state_csv(
    state_csv: str | Path,
    *,
    ts_col: str = "Created",
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    state_csv = Path(state_csv)
    s = pd.read_csv(state_csv, usecols=[ts_col])
    s[ts_col] = pd.to_datetime(s[ts_col], errors="coerce", utc=True)
    s = s.dropna(subset=[ts_col]).sort_values(ts_col)
    if s.empty:
        return None
    return s[ts_col].iloc[0], s[ts_col].iloc[-1]


def ranges_from_state_csv(
    state_csv: str | Path,
    *,
    ts_col: str = "Created",
    state_col: str = "state",
    target_state: str = "harbour",
    min_minutes: int = 5,
    merge_gap_minutes: int = 3,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Generic: reads state.csv and returns continuous intervals where state==target_state.
    """
    state_csv = Path(state_csv)
    s = pd.read_csv(state_csv)

    if ts_col not in s.columns:
        raise ValueError(f"'{ts_col}' not in {state_csv}. Columns: {list(s.columns)}")
    if state_col not in s.columns:
        raise ValueError(f"'{state_col}' not in {state_csv}. Columns: {list(s.columns)}")

    s[ts_col] = pd.to_datetime(s[ts_col], errors="coerce", utc=True)
    s = s.dropna(subset=[ts_col]).sort_values(ts_col)

    is_target = s[state_col].astype(str).str.lower().eq(target_state.lower())

    # Run-length encoding segments
    seg_id = is_target.ne(is_target.shift()).cumsum()
    segs = (
        s.assign(is_target=is_target, seg_id=seg_id)
         .groupby("seg_id", as_index=False)
         .agg(
            is_target=("is_target", "first"),
            start=(ts_col, "first"),
            end=(ts_col, "last"),
         )
    )
    segs = segs[segs["is_target"]].copy()
    if segs.empty:
        return []

    # Filter short segments
    dur_min = (segs["end"] - segs["start"]).dt.total_seconds() / 60.0
    segs = segs.loc[dur_min >= min_minutes].sort_values("start").reset_index(drop=True)
    if segs.empty:
        return []

    # Merge close segments
    gap = pd.Timedelta(minutes=merge_gap_minutes)
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    cur_s = segs.loc[0, "start"]
    cur_e = segs.loc[0, "end"]
    for i in range(1, len(segs)):
        s_i = segs.loc[i, "start"]
        e_i = segs.loc[i, "end"]
        if s_i <= cur_e + gap:
            cur_e = max(cur_e, e_i)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s_i, e_i
    merged.append((cur_s, cur_e))

    return merged

# -----------------------------
# Main plot builder
# -----------------------------
def build_csv_dropdown_plot(
    base_dir: str | Path,
    *,
    file_glob: str = "*.csv",
    max_traces: int = 60,             # safety for performance
    sample_every: int | None = None,  # e.g. 5 -> plot every 5th row (faster)

    # State shading input
    state_csv: str | Path | None = None,
    state_ts_col: str = "Created",
    state_col: str = "state",

    # Shading colors (tune alpha as you like)
    harbour_fill_rgba: str = "rgba(120, 120, 120, 0.4)",
    anchored_fill_rgba: str = "rgba(180, 180, 180, 0.4)",
    sailing_fill_rgba: str = "rgba(210, 210, 210, 0.4)",
):
    base_dir = Path(base_dir)
    csv_files = sorted(base_dir.glob(file_glob))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {base_dir}")

    # Prepare state-based ranges (only if state_csv provided)
    span: tuple[pd.Timestamp, pd.Timestamp] | None = None
    harbour_ranges_dt: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None
    anchored_ranges_dt: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None

    shading_requested = state_csv is not None
    if state_csv is not None:
        span = time_span_from_state_csv(state_csv, ts_col=state_ts_col)

        # These will return [] if state doesn't exist -> that's fine
        harbour_ranges_dt = ranges_from_state_csv(
            state_csv,
            ts_col=state_ts_col,
            state_col=state_col,
            target_state="harbour",
            min_minutes=5,
            merge_gap_minutes=3,
        )
        anchored_ranges_dt = ranges_from_state_csv(
            state_csv,
            ts_col=state_ts_col,
            state_col=state_col,
            target_state="anchored",
            min_minutes=5,
            merge_gap_minutes=3,
        )

    fig = go.Figure()
    trace_ranges: list[tuple[int, int, str]] = []  # (start_idx, end_idx_exclusive, label)
    any_datetime_x = False

    # --- Add data traces (all hidden initially) ---
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        ts_col = find_timestamp_col(df)
        if ts_col:
            x = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
            mask = x.notna()
            df = df.loc[mask].copy()
            x = x.loc[mask]
            any_datetime_x = True
        else:
            x = df.index

        if sample_every and sample_every > 1:
            df = df.iloc[::sample_every].copy()
            x = x[::sample_every]

        exclude = {c for c in ("Created", "Modified", "Inserted", "Timestamp", "Time") if c in df.columns}
        data_cols = [c for c in df.columns if c not in exclude]
        numeric_cols = [c for c in data_cols if pd.api.types.is_numeric_dtype(df[c])]

        if len(numeric_cols) > max_traces:
            numeric_cols = numeric_cols[:max_traces]

        start = len(fig.data)
        for col in numeric_cols:
            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=df[col],
                    mode="lines",
                    name=col,
                    visible=False,  # will become legendonly for selected dropdown item
                )
            )
        end = len(fig.data)
        trace_ranges.append((start, end, csv_path.stem))

    if len(fig.data) == 0:
        raise ValueError("No traces were added. Are the CSVs empty or non-numeric?")

    # --- Shading (base + overlays) ---
    if shading_requested:
        if not any_datetime_x:
            raise ValueError(
                "state_csv provided, but none of the CSVs had a usable timestamp column "
                "(Created/Modified/Inserted/Timestamp/Time)."
            )

        # Base layer: Sailing (whole span)
        if span is not None:
            t0, t1 = span
            fig.add_vrect(
                x0=t0,
                x1=t1,
                fillcolor=sailing_fill_rgba,
                line_width=0,
                layer="below",
            )

        # Overlay anchored (only if any)
        if anchored_ranges_dt:
            for s, e in anchored_ranges_dt:
                fig.add_vrect(
                    x0=s,
                    x1=e,
                    fillcolor=anchored_fill_rgba,
                    line_width=0,
                    layer="below",
                )

        # Overlay harbour (only if any)
        if harbour_ranges_dt:
            for s, e in harbour_ranges_dt:
                fig.add_vrect(
                    x0=s,
                    x1=e,
                    fillcolor=harbour_fill_rgba,
                    line_width=0,
                    layer="below",
                )

        # ✅ State labels ABOVE plot (annotations)
        state_items = [
            ("Sailing", sailing_fill_rgba),
            ("Anchored", anchored_fill_rgba),
            ("Harbour", harbour_fill_rgba),
        ]
        show_anchored = bool(anchored_ranges_dt)

        # layout in "paper" coordinates (0..1)
        x0 = 0.2  # starting x (left)
        y0 = 1.08  # y above plot
        sw = 0.018  # swatch width
        sh = 0.03  # swatch height
        gap = 0.12  # horizontal spacing between items
        txt_dx = 0.022  # label offset to the right of swatch

        j = 0
        for label, color in state_items:
            if label == "Anchored" and not show_anchored:
                continue

            # colored square (shape)
            fig.add_shape(
                type="rect",
                xref="paper", yref="paper",
                x0=x0 + j * gap,
                x1=x0 + j * gap + sw,
                y0=y0,
                y1=y0 + sh,
                fillcolor=color,
                line_width=0,
                layer="above",  # show above plot area
            )

            # text label
            fig.add_annotation(
                xref="paper", yref="paper",
                x=x0 + j * gap + txt_dx,
                y=y0 + sh / 2,
                text=label,
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=13),
            )

            j += 1
    # --- Dropdown buttons ---
    n_traces = len(fig.data)
    buttons = []

    for start, end, label in trace_ranges:
        visible = [False] * n_traces

        # Selected subsystem: show signals in legend only (user clicks to display)
        for i in range(start, end):
            visible[i] = "legendonly"

        buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": visible},
                    {"title": f"Subsystem CSV: {label}  (signals: {end - start})"},
                ],
            )
        )

    # Default: no subsystem selected -> user must pick from dropdown
    default_title = "Subsystem CSV (select from dropdown)"

    fig.update_layout(
        title=default_title,
        xaxis_title="Time" if any_datetime_x else "Index",
        xaxis_title_standoff=2,
        yaxis_title="Value",

        updatemenus=[dict(buttons=buttons, direction="down", x=0.01, y=1.15, showactive=True)],

        # Legend below plot (signals only)
        legend=dict(
            itemsizing="constant",
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0.0,
        ),

        # More top margin to make room for state labels above plot
        margin=dict(l=20, r=20, t=170, b=140),

        height=850,

        legend_itemclick="toggle",
        legend_itemdoubleclick="toggleothers",
    )

    return fig


def main():
    output_dir = cfg.DATA_PATH / "plotly_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_num = build_csv_dropdown_plot(
        base_dir=cfg.DATA_PATH / "subsystems" / "DS1" / "NUMERIC",
        state_csv=cfg.DATA_PATH / "subsystems" / "DS1"/"state.csv",
        state_ts_col="Created",
        state_col="state",
        max_traces=60,
    )
    fig_num.write_html(f"DS{1}_plot.html", auto_open=False)

    fig_num = build_csv_dropdown_plot(
        base_dir=cfg.DATA_PATH / "subsystems" / "DS2" / "NUMERIC",
        state_csv=cfg.DATA_PATH / "subsystems" / "DS2"/"state.csv",
        state_ts_col="Created",
        state_col="state",
        max_traces=60,
    )
    fig_num.write_html(f"DS{2}_plot.html", auto_open=False)

if __name__ == "__main__":
    main()
