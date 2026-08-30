#!/usr/bin/env python3
"""Replace slides 13-18 with fixed-seed random-vs-trained test results."""

import csv
import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_DIR, "Models", "retest_results", "20260828_231500")
RESULTS_CSV = os.path.join(RESULTS_DIR, "comparison.csv")
SOURCE_PPTX = "/home/tnmfdrt/Downloads/Optimizing_Traffic_Control_ With_A_Deep_Q_Network-FINAL(1).pptx"
OUTPUT_PPTX = "/home/tnmfdrt/Downloads/Optimizing_Traffic_Control_ With_A_Deep_Q_Network-RETEST-UPDATED.pptx"

BG = "104862"
PANEL = "0C3C52"
GRID = "47788B"
TEXT = "E7F0F3"
MUTED = "C5D6DD"
TITLE = "FF0000"
PYTORCH = "92D050"
NUMPY = "FFF200"
NEGATIVE = "FF6B6B"
ACCENT = "42C6E8"


def rgb(hex_color):
    return RGBColor.from_string(hex_color)


def load_results():
    with open(RESULTS_CSV, newline="", encoding="utf-8") as result_file:
        rows = list(csv.DictReader(result_file))

    results = {}
    for row in rows:
        before = float(row["Before Random"])
        after = float(row["After Trained"])
        results[row["Configuration"]] = {
            "before": before,
            "after": after,
            "delta": after - before,
            "relative": (after - before) / abs(before) * 100.0,
        }
    return results


def result_for(results, scenario, backend, mode):
    return results[f"{scenario} | {backend} | {mode}"]


def style_axis(axis):
    axis.set_facecolor(f"#{BG}")
    axis.tick_params(colors=f"#{MUTED}", labelsize=10)
    axis.xaxis.label.set_color(f"#{MUTED}")
    axis.yaxis.label.set_color(f"#{MUTED}")
    axis.title.set_color(f"#{TEXT}")
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.grid(axis="x", color=f"#{GRID}", alpha=0.55, linewidth=0.8)
    axis.set_axisbelow(True)


def make_delta_chart(results, scenario, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), facecolor=f"#{BG}")
    colors = [f"#{PYTORCH}", f"#{NUMPY}"]
    backends = ("PyTorch", "NumPy")

    for axis, mode in zip(axes, ("Standard", "Zhang-style")):
        records = [result_for(results, scenario, backend, mode) for backend in backends]
        deltas = [record["delta"] for record in records]
        positions = np.array([1, 0])
        axis.barh(positions, deltas, color=colors, height=0.48, zorder=3)
        style_axis(axis)
        axis.axvline(0, color=f"#{TEXT}", linewidth=1.5, zorder=4)

        if min(deltas) >= 0:
            maximum = max(deltas)
            axis.set_xlim(-maximum * 0.035, maximum * 1.30)
        elif max(deltas) <= 0:
            magnitude = abs(min(deltas))
            axis.set_xlim(-magnitude * 1.30, magnitude * 0.12)
            axis.axvspan(axis.get_xlim()[0], 0, color=f"#{NEGATIVE}", alpha=0.08)
        else:
            span = max(deltas) - min(deltas)
            axis.set_xlim(min(deltas) - span * 0.18, max(deltas) + span * 0.18)

        x_min, x_max = axis.get_xlim()
        x_span = x_max - x_min
        labels = []
        for backend, record in zip(backends, records):
            labels.append(
                f"{backend}\n{record['before']:,.2f}  →  {record['after']:,.2f}"
            )

        axis.set_yticks(positions, labels)
        axis.set_ylim(-0.65, 1.65)
        axis.set_title(
            "Standard reward" if mode == "Standard" else "Zhang-style reward",
            fontsize=15,
            fontweight="bold",
            pad=14,
        )
        axis.set_xlabel("Change in total reward  (trained − random)", fontsize=10)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))

        for position, delta in zip(positions, deltas):
            if delta >= 0:
                x_value = delta + x_span * 0.025
                alignment = "left"
            else:
                x_value = delta - x_span * 0.025
                alignment = "right"
            axis.text(
                x_value,
                position,
                f"{delta:+,.2f}",
                va="center",
                ha=alignment,
                color=f"#{TEXT}" if delta >= 0 else f"#{NEGATIVE}",
                fontsize=12,
                fontweight="bold",
            )

        axis.text(
            0.98,
            0.04,
            "Random baseline = 0",
            transform=axis.transAxes,
            ha="right",
            color=f"#{MUTED}",
            fontsize=9,
        )

    figure.subplots_adjust(left=0.125, right=0.985, top=0.86, bottom=0.19, wspace=0.42)
    figure.savefig(output_path, dpi=220, facecolor=figure.get_facecolor())
    plt.close(figure)


def make_scale_chart(results, mode, output_path):
    scenarios = ("Single", "2x2", "4x4")
    scenario_labels = ("Single\n1 agent", "2×2\n4 agents", "4×4\n16 agents")
    backends = ("PyTorch", "NumPy")
    colors = (f"#{PYTORCH}", f"#{NUMPY}")

    values = {
        backend: [result_for(results, scenario, backend, mode)["relative"] for scenario in scenarios]
        for backend in backends
    }

    figure, axis = plt.subplots(figsize=(13.0, 5.0), facecolor=f"#{BG}")
    style_axis(axis)
    positions = np.arange(len(scenarios))
    width = 0.31

    for offset, backend, color in zip((-width / 2, width / 2), backends, colors):
        bars = axis.bar(
            positions + offset,
            values[backend],
            width,
            label=backend,
            color=color,
            zorder=3,
        )
        for bar, value in zip(bars, values[backend]):
            if value >= 0:
                y_value = value + 1.6
                vertical = "bottom"
                color_value = f"#{TEXT}"
            else:
                y_value = value - 1.3
                vertical = "top"
                color_value = f"#{NEGATIVE}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                y_value,
                f"{value:+.1f}%",
                ha="center",
                va=vertical,
                color=color_value,
                fontsize=12,
                fontweight="bold",
            )

    axis.axhline(0, color=f"#{TEXT}", linewidth=1.4, zorder=4)
    if mode == "Standard":
        axis.set_ylim(0, 112)
        axis.set_yticks(np.arange(0, 101, 20))
    else:
        axis.set_ylim(-15, 44)
        axis.set_yticks(np.arange(-10, 41, 10))
        axis.axhspan(-15, 0, color=f"#{NEGATIVE}", alpha=0.08)

    axis.set_xticks(positions, scenario_labels)
    axis.tick_params(axis="x", labelsize=12, pad=8)
    axis.set_ylabel("Reward change relative to random baseline (%)", fontsize=11)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    axis.grid(axis="y", color=f"#{GRID}", alpha=0.55, linewidth=0.8)
    axis.grid(axis="x", visible=False)
    legend = axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    for label in legend.get_texts():
        label.set_color(f"#{TEXT}")

    axis.text(
        0.99,
        0.02,
        "Positive = trained model earned higher reward",
        transform=axis.transAxes,
        ha="right",
        color=f"#{MUTED}",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.09, right=0.985, top=0.83, bottom=0.19)
    figure.savefig(output_path, dpi=220, facecolor=figure.get_facecolor())
    plt.close(figure)


def clear_slide(slide):
    for shape in list(slide.shapes):
        slide.shapes._spTree.remove(shape._element)


def add_textbox(
    slide,
    x,
    y,
    width,
    height,
    text,
    font_size,
    color=TEXT,
    bold=False,
    font_name="Aptos",
    alignment=PP_ALIGN.LEFT,
    vertical=MSO_ANCHOR.MIDDLE,
    margin=0.03,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    box.text_frame.clear()
    box.text_frame.margin_left = Inches(margin)
    box.text_frame.margin_right = Inches(margin)
    box.text_frame.margin_top = Inches(margin)
    box.text_frame.margin_bottom = Inches(margin)
    box.text_frame.vertical_anchor = vertical
    paragraph = box.text_frame.paragraphs[0]
    paragraph.alignment = alignment
    run = paragraph.add_run()
    run.text = text
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    return box


def add_title(slide, text):
    return add_textbox(
        slide,
        0.55,
        0.22,
        12.15,
        0.66,
        text,
        25.5,
        color=TITLE,
        font_name="Aptos Display",
    )


def add_subtitle(slide, text):
    return add_textbox(slide, 0.58, 0.86, 12.1, 0.34, text, 11.5, color=MUTED)


def add_callout(slide, text, color=TEXT, font_size=11.5):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.72),
        Inches(6.34),
        Inches(11.90),
        Inches(0.70),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(PANEL)
    shape.line.color.rgb = rgb(GRID)
    shape.line.width = Pt(1.0)
    shape.text_frame.clear()
    shape.text_frame.margin_left = Inches(0.16)
    shape.text_frame.margin_right = Inches(0.16)
    shape.text_frame.margin_top = Inches(0.03)
    shape.text_frame.margin_bottom = Inches(0.03)
    shape.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = shape.text_frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER
    run = paragraph.add_run()
    run.text = text
    run.font.name = "Aptos"
    run.font.size = Pt(font_size)
    run.font.bold = True
    run.font.color.rgb = rgb(color)
    return shape


def add_chart_slide(slide, title, subtitle, image_path, callout, callout_color=TEXT):
    clear_slide(slide)
    add_title(slide, title)
    add_subtitle(slide, subtitle)
    slide.shapes.add_picture(
        image_path,
        Inches(0.55),
        Inches(1.25),
        width=Inches(12.25),
        height=Inches(5.02),
    )
    add_callout(slide, callout, callout_color)


def add_card(slide, x, y, width, height, heading, body, accent_color):
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(x),
        Inches(y),
        Inches(width),
        Inches(height),
    )
    card.fill.solid()
    card.fill.fore_color.rgb = rgb(PANEL)
    card.line.color.rgb = rgb(accent_color)
    card.line.width = Pt(1.4)
    card.text_frame.clear()
    card.text_frame.margin_left = Inches(0.16)
    card.text_frame.margin_right = Inches(0.16)
    card.text_frame.margin_top = Inches(0.09)
    card.text_frame.margin_bottom = Inches(0.08)
    card.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE

    heading_paragraph = card.text_frame.paragraphs[0]
    heading_paragraph.alignment = PP_ALIGN.CENTER
    heading_run = heading_paragraph.add_run()
    heading_run.text = heading
    heading_run.font.name = "Aptos Display"
    heading_run.font.size = Pt(23)
    heading_run.font.bold = True
    heading_run.font.color.rgb = rgb(accent_color)

    body_paragraph = card.text_frame.add_paragraph()
    body_paragraph.alignment = PP_ALIGN.CENTER
    body_run = body_paragraph.add_run()
    body_run.text = body
    body_run.font.name = "Aptos"
    body_run.font.size = Pt(10.5)
    body_run.font.color.rgb = rgb(TEXT)
    return card


def add_finding_box(slide, x, heading, lines, accent_color):
    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(x),
        Inches(3.38),
        Inches(5.88),
        Inches(1.98),
    )
    box.fill.solid()
    box.fill.fore_color.rgb = rgb(PANEL)
    box.line.color.rgb = rgb(GRID)
    box.line.width = Pt(1.0)
    box.text_frame.clear()
    box.text_frame.margin_left = Inches(0.22)
    box.text_frame.margin_right = Inches(0.18)
    box.text_frame.margin_top = Inches(0.12)
    box.text_frame.vertical_anchor = MSO_ANCHOR.TOP

    heading_paragraph = box.text_frame.paragraphs[0]
    heading_run = heading_paragraph.add_run()
    heading_run.text = heading
    heading_run.font.name = "Aptos Display"
    heading_run.font.size = Pt(16)
    heading_run.font.bold = True
    heading_run.font.color.rgb = rgb(accent_color)

    for line in lines:
        paragraph = box.text_frame.add_paragraph()
        paragraph.level = 0
        paragraph.text = f"• {line}"
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(10.7)
        paragraph.font.color.rgb = rgb(TEXT)
        paragraph.space_before = Pt(4)
    return box


def build_conclusion_slide(slide):
    clear_slide(slide)
    add_title(slide, "Main Conclusions")
    add_subtitle(
        slide,
        "Fixed seed 42 | Random greedy policy → trained greedy policy | 3,600 simulated seconds per evaluation",
    )

    add_card(slide, 0.68, 1.35, 3.72, 1.70, "10 / 12", "trained models beat the random baseline", PYTORCH)
    add_card(slide, 4.80, 1.35, 3.72, 1.70, "≈99%", "random penalty removed by standard single and 4×4 models", ACCENT)
    add_card(slide, 8.92, 1.35, 3.72, 1.70, "4×4 Zhang", "−2.6% PyTorch | −8.6% NumPy versus random", NEGATIVE)

    add_finding_box(
        slide,
        0.68,
        "What scaled successfully",
        [
            "Standard reward beats random at every scale; both backends remove ≈99% of the 4×4 random penalty.",
            "Full local observations and waiting-time feedback remain learnable under heavier traffic.",
        ],
        PYTORCH,
    )
    add_finding_box(
        slide,
        6.78,
        "Why Zhang-style regressed at 4×4",
        [
            "16 independent agents face ≈7× the attempted inflow of 2×2, making local control non-stationary.",
            "Lane visibility and detected vehicles are independently resampled at 60%, producing noisy state/reward pairs.",
            "Only 3 episodes were run; ε reached its floor by episode 2 and the final—not best—checkpoint was saved.",
        ],
        NUMPY,
    )

    takeaway = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.68),
        Inches(5.62),
        Inches(11.96),
        Inches(1.15),
    )
    takeaway.fill.solid()
    takeaway.fill.fore_color.rgb = rgb("123B4C")
    takeaway.line.color.rgb = rgb(ACCENT)
    takeaway.line.width = Pt(1.6)
    takeaway.text_frame.clear()
    takeaway.text_frame.margin_left = Inches(0.28)
    takeaway.text_frame.margin_right = Inches(0.28)
    takeaway.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = takeaway.text_frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER
    run = paragraph.add_run()
    run.text = (
        "Interpretation: Zhang’s 4×4 checkpoint did not maximize speed, but the random policy also admitted fewer "
        "vehicles. Judge traffic control with reward + throughput + waiting/stops—not reward alone."
    )
    run.font.name = "Aptos"
    run.font.size = Pt(13.5)
    run.font.bold = True
    run.font.color.rgb = rgb(TEXT)


def main():
    results = load_results()
    presentation = Presentation(SOURCE_PPTX)

    with tempfile.TemporaryDirectory(prefix="cs615_retest_slides_") as asset_dir:
        single_chart = os.path.join(asset_dir, "single_delta.png")
        grid_2x2_chart = os.path.join(asset_dir, "2x2_delta.png")
        grid_4x4_chart = os.path.join(asset_dir, "4x4_delta.png")
        standard_scale_chart = os.path.join(asset_dir, "standard_scale.png")
        zhang_scale_chart = os.path.join(asset_dir, "zhang_scale.png")

        make_delta_chart(results, "Single", single_chart)
        make_delta_chart(results, "2x2", grid_2x2_chart)
        make_delta_chart(results, "4x4", grid_4x4_chart)
        make_scale_chart(results, "Standard", standard_scale_chart)
        make_scale_chart(results, "Zhang-style", zhang_scale_chart)

        add_chart_slide(
            presentation.slides[12],
            "Test Results – Single Intersection",
            "Random policy is the reference (ΔR = 0):  ΔR = trained reward − random reward.  Positive is better.",
            single_chart,
            "Both implementations improve. Standard nearly removes the random penalty; Zhang-style gains are smaller.",
        )
        add_chart_slide(
            presentation.slides[13],
            "Test Results – 2×2 Grid",
            "Random policy is the reference (ΔR = 0):  ΔR = trained reward − random reward.  Positive is better.",
            grid_2x2_chart,
            "All four models beat random; Zhang-style gains are similar, while standard NumPy improves less than PyTorch.",
        )
        add_chart_slide(
            presentation.slides[14],
            "Test Results – 4×4 Grid",
            "Random policy is the reference (ΔR = 0):  ΔR = trained reward − random reward.  Positive is better.",
            grid_4x4_chart,
            "Why Zhang falls: 16 agents, ≈7× 2×2 inflow, stochastic 60% detection, and only 3 episodes; the final checkpoint had not converged.",
            NEGATIVE,
        )
        add_chart_slide(
            presentation.slides[15],
            "Scale Effect – Standard Reward",
            "Relative change = (trained − random) ÷ |random| × 100.  A 100% gain moves a negative baseline to zero.",
            standard_scale_chart,
            "PyTorch removes about 99% of the random penalty at every scale; NumPy's only major dip is the 2×2 grid.",
        )
        add_chart_slide(
            presentation.slides[16],
            "Scale Effect – Zhang-Style Reward",
            "Relative change = (trained − random) ÷ |random| × 100.  Positive is better; negative is regression.",
            zhang_scale_chart,
            "At 16 agents both backends regress; the speed-only score may also favor random policies that block and skip more departures.",
            NEGATIVE,
        )
        build_conclusion_slide(presentation.slides[17])

        presentation.save(OUTPUT_PPTX)

    print(OUTPUT_PPTX)


if __name__ == "__main__":
    main()
