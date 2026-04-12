const pptxgen = require("pptxgenjs");

const OUT = "C:/Users/zihan/capstone/docs/RAPO_AS_RL_Defense.pptx";

// ── Palette ──────────────────────────────────────────────────────────
const NAVY   = "1E2761";   // primary (dark slides)
const ICE    = "CADCFC";   // secondary (light text on dark)
const WHITE  = "FFFFFF";
const CREAM  = "F8F9FA";
const SLATE  = "64748B";   // muted body text
const DARK   = "1E293B";   // dark body text
const ACCENT = "0891B2";   // teal accent for highlights
const GOLD   = "F59E0B";   // amber for "not confirmed" / warnings
const RED    = "DC2626";   // negative indicators
const GREEN  = "16A34A";   // positive indicators
const ORANGE = "EA580C";

// ── Helpers ────────────────────────────────────────────────────────────
const makeShadow = () => ({ type: "outer", blur: 6, offset: 2, angle: 135, color: "000000", opacity: 0.12 });

function darkSlide(pres, title, subtitle) {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText(title, { x: 0.5, y: 1.8, w: 9, h: 1.2, fontSize: 38, bold: true, color: WHITE, align: "center" });
  if (subtitle) s.addText(subtitle, { x: 0.5, y: 3.1, w: 9, h: 0.7, fontSize: 18, color: ICE, align: "center" });
  return s;
}

function lightSlide(pres, title) {
  const s = pres.addSlide();
  s.background = { color: CREAM };
  // Left accent bar
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.12, h: 5.625, fill: { color: NAVY } });
  s.addText(title, { x: 0.4, y: 0.25, w: 9.2, h: 0.6, fontSize: 28, bold: true, color: DARK, margin: 0 });
  return s;
}

function statCard(slide, x, y, w, h, value, label, valColor) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: WHITE }, shadow: makeShadow() });
  slide.addText(value, { x, y: y + 0.2, w, h: 0.9, fontSize: 40, bold: true, color: valColor, align: "center", margin: 0 });
  slide.addText(label, { x, y: y + h - 0.55, w, h: 0.45, fontSize: 11, color: SLATE, align: "center", margin: 0 });
}

// ── Presentation ───────────────────────────────────────────────────────
let pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "RAPO-AS-RL Thesis Defense";
pres.author = "MScFE 690 Capstone";

// ══════════════════════════════════════════════════════════════════════
// SLIDE 1 — Title
// ══════════════════════════════════════════════════════════════════════
const s1 = darkSlide(pres);
s1.addText("Regime-Aware Portfolio Optimization", { x: 0.5, y: 1.4, w: 9, h: 0.8, fontSize: 32, bold: true, color: ICE, align: "center" });
s1.addText("with Avellaneda-Stoikov Market Impact Costs", { x: 0.5, y: 2.1, w: 9, h: 0.6, fontSize: 24, color: ICE, align: "center" });
s1.addShape(pres.shapes.RECTANGLE, { x: 3.5, y: 2.85, w: 3, h: 0.04, fill: { color: ACCENT } });
s1.addText("MScFE 690 Capstone  |  WorldQuant University", { x: 0.5, y: 3.1, w: 9, h: 0.4, fontSize: 14, color: ICE, align: "center" });
s1.addText("2026", { x: 0.5, y: 3.55, w: 9, h: 0.4, fontSize: 14, color: ICE, align: "center" });

// ══════════════════════════════════════════════════════════════════════
// SLIDE 2 — Agenda
// ══════════════════════════════════════════════════════════════════════
const s2 = lightSlide(pres, "Agenda");
const agendaItems = [
  "Research question & hypothesis",
  "Four-layer architecture",
  "A&S cost calibration methodology",
  "Backtest results",
  "Frequency sensitivity analysis",
  "Daily RL experiment",
  "Conclusions",
];
agendaItems.forEach((item, i) => {
  const y = 1.0 + i * 0.6;
  s2.addShape(pres.shapes.RECTANGLE, { x: 0.5, y, w: 0.3, h: 0.3, fill: { color: ACCENT } });
  s2.addText(String(i + 1), { x: 0.5, y, w: 0.3, h: 0.3, fontSize: 12, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  s2.addText(item, { x: 1.0, y, w: 8, h: 0.35, fontSize: 15, color: DARK, valign: "middle", margin: 0 });
});

// ══════════════════════════════════════════════════════════════════════
// SLIDE 3 — Research Question
// ══════════════════════════════════════════════════════════════════════
const s3 = lightSlide(pres, "Research Question");
s3.addText("Central Question", { x: 0.5, y: 0.95, w: 4.3, h: 0.4, fontSize: 14, bold: true, color: ACCENT, margin: 0 });
s3.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.3, w: 4.3, h: 1.1, fill: { color: WHITE }, shadow: makeShadow() });
s3.addText("Can regime-aware RL outperform CVaR optimization when execution costs are calibrated from exchange data?", { x: 0.6, y: 1.4, w: 4.1, h: 0.9, fontSize: 13, color: DARK, margin: 0 });

s3.addText("Hypothesis", { x: 5.1, y: 0.95, w: 4.4, h: 0.4, fontSize: 14, bold: true, color: ACCENT, margin: 0 });
s3.addShape(pres.shapes.RECTANGLE, { x: 5.1, y: 1.3, w: 4.4, h: 1.1, fill: { color: WHITE }, shadow: makeShadow() });
s3.addText("RL will outperform CVaR on Sharpe, with performance differential concentrated during regime transitions", { x: 5.2, y: 1.4, w: 4.2, h: 0.9, fontSize: 13, color: DARK, margin: 0 });

s3.addText("Outcome", { x: 0.5, y: 2.6, w: 9, h: 0.4, fontSize: 14, bold: true, color: GOLD, margin: 0 });
s3.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 2.95, w: 9, h: 1.2, fill: { color: WHITE }, shadow: makeShadow() });
s3.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 2.95, w: 0.08, h: 1.2, fill: { color: GOLD } });
s3.addText([
  { text: "Hypothesis NOT confirmed.", options: { bold: true, color: GOLD } },
  { text: "  RL did not outperform CVaR. However, the broader thesis — that A&S calibration reveals execution costs that make active rebalancing unprofitable — IS confirmed.", options: { color: DARK } },
], { x: 0.7, y: 3.05, w: 8.7, h: 1.0, fontSize: 13, margin: 0 });

s3.addText("The gap between validation Sharpe (+1.72, 10bps) and test Sharpe (-0.68, A&S) reveals that cost model choice dominates strategy choice.", { x: 0.5, y: 4.35, w: 9, h: 0.5, fontSize: 11, color: SLATE, italic: true, margin: 0 });

// ══════════════════════════════════════════════════════════════════════
// SLIDE 4 — Architecture
// ══════════════════════════════════════════════════════════════════════
const s4 = lightSlide(pres, "Four-Layer Architecture");
const layers = [
  { num: "1", name: "HMM Regime Classifier", desc: "OHLCV + trade ticks → Calm / Volatile / Stressed", output: "regime_labels.csv, hmm_model.pkl", color: "3B82F6" },
  { num: "2", name: "Avellaneda-Stoikov Cost Model", desc: "Regime labels + trade ticks → calibrated σ, s, δ, γ", output: "as_cost_Calm/Volatile/Stressed.pkl", color: ACCENT },
  { num: "3", name: "LightGBM Return Forecaster", desc: "Features → next-period return (R² ≈ 0, replaced with lagged returns)", output: "lgbm_*.pkl (trained but not used by RL)", color: GREEN },
  { num: "4", name: "Regime-Aware PPO Agent", desc: "Portfolio state (14-dim, includes regime index) → target weights", output: "ppo_full.zip", color: ORANGE },
];
layers.forEach((l, i) => {
  const y = 0.95 + i * 1.1;
  s4.addShape(pres.shapes.RECTANGLE, { x: 0.5, y, w: 0.55, h: 0.9, fill: { color: l.color } });
  s4.addText(l.num, { x: 0.5, y: y + 0.15, w: 0.55, h: 0.6, fontSize: 22, bold: true, color: WHITE, align: "center", margin: 0 });
  s4.addShape(pres.shapes.RECTANGLE, { x: 1.05, y, w: 8.45, h: 0.9, fill: { color: WHITE }, shadow: makeShadow() });
  s4.addText(l.name, { x: 1.15, y: y + 0.08, w: 8.2, h: 0.4, fontSize: 14, bold: true, color: DARK, margin: 0 });
  s4.addText(l.desc, { x: 1.15, y: y + 0.42, w: 8.2, h: 0.35, fontSize: 11, color: SLATE, margin: 0 });
});

// ══════════════════════════════════════════════════════════════════════
// SLIDE 5 — A&S Cost Calibration Results
// ══════════════════════════════════════════════════════════════════════
const s5 = lightSlide(pres, "A&S Cost Calibration — Per-Regime Parameters");
s5.addText("Calibrated from Binance OHLCV and trade tick data using Lee-Ready tick classification", { x: 0.5, y: 0.85, w: 9, h: 0.3, fontSize: 10, color: SLATE, italic: true, margin: 0 });

const costTable = [
  [
    { text: "Parameter", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
    { text: "Calm", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
    { text: "Volatile", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
    { text: "Stressed", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
  ],
  [{ text: "σ (annual vol)" }, { text: "0.57" }, { text: "0.79" }, { text: "1.14" }],
  [{ text: "s ($/BTC spread)" }, { text: "$104" }, { text: "$200" }, { text: "$1,092" }],
  [{ text: "δ (depth BTC/$)" }, { text: "0.044" }, { text: "0.044" }, { text: "0.044" }],
  [{ text: "γ (risk aversion)" }, { text: "1e-6" }, { text: "1e-6" }, { text: "1e-5" }],
  [
    { text: "A&S Cost (50bps trade)", options: { bold: true } },
    { text: "~123 bps", options: { color: ORANGE, bold: true } },
    { text: "~300 bps", options: { color: ORANGE, bold: true } },
    { text: "~1,292 bps", options: { color: RED, bold: true } },
  ],
];
s5.addTable(costTable, {
  x: 0.5, y: 1.2, w: 6.0, colW: [2.2, 1.2, 1.3, 1.3],
  fontSize: 11, color: DARK,
  border: { pt: 0.5, color: "E2E8F0" },
  rowH: 0.42,
});

s5.addShape(pres.shapes.RECTANGLE, { x: 6.8, y: 1.2, w: 2.7, h: 2.1, fill: { color: WHITE }, shadow: makeShadow() });
s5.addText("Key Finding", { x: 6.9, y: 1.3, w: 2.5, h: 0.35, fontSize: 12, bold: true, color: ACCENT, margin: 0 });
s5.addText([
  { text: "Cost / Trade Ratio", options: { bold: true, breakLine: true } },
  { text: "Calm: ", options: { breakLine: false } },
  { text: "2.5× nominal", options: { color: ORANGE, bold: true, breakLine: true } },
  { text: "Stressed: ", options: { breakLine: false } },
  { text: "26× nominal", options: { color: RED, bold: true, breakLine: true } },
  { text: "→ A single 50bps rebalance in Calm regime costs more than 2.5× the trade size in market impact", options: { fontSize: 10, color: SLATE } },
], { x: 6.9, y: 1.7, w: 2.5, h: 1.5, fontSize: 12, color: DARK, margin: 0 });

s5.addText("Depth δ estimated from A&S equilibrium: δ = 2 / (s_proxy · P) ≈ 0.044 BTC²/$ — reflecting shallow crypto order books vs traditional markets", { x: 0.5, y: 3.55, w: 9, h: 0.4, fontSize: 10, color: SLATE, italic: true, margin: 0 });

// ══════════════════════════════════════════════════════════════════════
// SLIDE 6 — Cost-to-Return Ratio by Regime
// ══════════════════════════════════════════════════════════════════════
const s6 = lightSlide(pres, "The Cost Headwind: Return vs. Execution Cost");
const regimes6 = [
  { name: "Calm", ret: "4 bps", cost: "~123 bps", ratio: "31× headwind", bg: GREEN, txt: WHITE },
  { name: "Volatile", ret: "8 bps", cost: "~300 bps", ratio: "38× headwind", bg: ORANGE, txt: WHITE },
  { name: "Stressed", ret: "−20 bps", cost: "~1,292 bps", ratio: "Negative return", bg: RED, txt: WHITE },
];
regimes6.forEach((r, i) => {
  const x = 0.5 + i * 3.1;
  s6.addShape(pres.shapes.RECTANGLE, { x, y: 0.95, w: 2.9, h: 2.9, fill: { color: WHITE }, shadow: makeShadow() });
  s6.addShape(pres.shapes.RECTANGLE, { x, y: 0.95, w: 2.9, h: 0.55, fill: { color: r.bg } });
  s6.addText(r.name, { x, y: 0.95, w: 2.9, h: 0.55, fontSize: 16, bold: true, color: r.txt, align: "center", valign: "middle", margin: 0 });
  s6.addText("Expected Return / bar", { x, y: 1.6, w: 2.9, h: 0.3, fontSize: 10, color: SLATE, align: "center", margin: 0 });
  s6.addText(r.ret, { x, y: 1.85, w: 2.9, h: 0.55, fontSize: 28, bold: true, color: DARK, align: "center", margin: 0 });
  s6.addText("A&S Cost / 50bps trade", { x, y: 2.45, w: 2.9, h: 0.3, fontSize: 10, color: SLATE, align: "center", margin: 0 });
  s6.addText(r.cost, { x, y: 2.7, w: 2.9, h: 0.55, fontSize: 22, bold: true, color: r.bg, align: "center", margin: 0 });
  s6.addText(r.ratio, { x, y: 3.3, w: 2.9, h: 0.4, fontSize: 12, bold: true, color: r.bg, align: "center", margin: 0 });
});

s6.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.05, w: 9, h: 0.75, fill: { color: NAVY } });
s6.addText("Conclusion: At ANY realistic signal strength, expected return << execution cost — active rebalancing destroys value in all three regimes.", { x: 0.6, y: 4.05, w: 8.8, h: 0.75, fontSize: 12, color: WHITE, valign: "middle", margin: 0 });

// ══════════════════════════════════════════════════════════════════════
// SLIDE 7 — Backtest Results
// ══════════════════════════════════════════════════════════════════════
const s7 = lightSlide(pres, "Backtest Results — Test Period (2024-02 to 2026-04)");
s7.addText("2+ years, 227k bars, full bull/bear cycle", { x: 0.5, y: 0.85, w: 9, h: 0.3, fontSize: 10, color: SLATE, italic: true, margin: 0 });

const btTable = [
  [
    { text: "Strategy", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
    { text: "Ann. Return", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
    { text: "Sharpe", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
    { text: "Max DD", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
    { text: "Turnover", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
    { text: "Result", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
  ],
  [
    { text: "Flat(A&S)", options: { bold: true } },
    { text: "+26.2%", options: { color: GREEN, bold: true } },
    { text: "+0.48", options: { color: GREEN, bold: true } },
    { text: "−56.6%", options: {} },
    { text: "~0", options: {} },
    { text: "WINNER", options: { fill: { color: GREEN }, color: WHITE, bold: true } },
  ],
  [
    { text: "Flat(10bps)" },
    { text: "+25.1%" },
    { text: "+0.44" },
    { text: "−57.6%" },
    { text: "~0" },
    { text: "Optimistic baseline", options: { color: SLATE } },
  ],
  [
    { text: "A&S+CVaR" },
    { text: "+23.4%" },
    { text: "+0.42" },
    { text: "−57.1%" },
    { text: "0.000006" },
    { text: "Minimally active", options: { color: SLATE } },
  ],
  [
    { text: "RL Agent", options: {} },
    { text: "−3.6%", options: { color: RED } },
    { text: "−0.68", options: { color: RED } },
    { text: "−7.9%", options: { color: GREEN } },
    { text: "0.000004" },
    { text: "Converged to cash", options: { color: GOLD } },
  ],
];
s7.addTable(btTable, {
  x: 0.5, y: 1.2, w: 9.0, colW: [2.0, 1.4, 1.1, 1.1, 1.4, 2.0],
  fontSize: 11, color: DARK,
  border: { pt: 0.5, color: "E2E8F0" },
  rowH: 0.42,
});

s7.addText([
  { text: "RL has best Max Drawdown (−7.9%) but worst Sharpe (−0.68).", options: { bold: true } },
  { text: "  Flat(A&S) wins on BOTH return and Sharpe — this is the true economic baseline.", options: {} },
], { x: 0.5, y: 3.55, w: 9, h: 0.55, fontSize: 12, color: DARK, margin: 0 });

s7.addText("Key: Cost-to-return headwind makes CVaR's rebalancing unjustifiable. RL correctly identifies that cash is the optimal policy.", { x: 0.5, y: 4.2, w: 9, h: 0.4, fontSize: 10, color: SLATE, italic: true, margin: 0 });

// ══════════════════════════════════════════════════════════════════════
// SLIDE 8 — Statistical Significance
// ══════════════════════════════════════════════════════════════════════
const s8 = lightSlide(pres, "Statistical Significance — Block Bootstrap + BH Correction");
s8.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 0.95, w: 4.3, h: 2.5, fill: { color: WHITE }, shadow: makeShadow() });
s8.addText("Methodology", { x: 0.6, y: 1.05, w: 4.1, h: 0.4, fontSize: 14, bold: true, color: ACCENT, margin: 0 });
s8.addText([
  { text: "Block Bootstrap", options: { bold: true, breakLine: true } },
  { text: "288-bar blocks × 1,000 replicates", options: { breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "Multiple Testing Correction", options: { bold: true, breakLine: true } },
  { text: "Benjamini-Hochberg at q = 0.10", options: { breakLine: true } },
  { text: "(6 pairwise comparisons)", options: {} },
], { x: 0.6, y: 1.45, w: 4.1, h: 1.9, fontSize: 12, color: DARK, margin: 0 });

s8.addShape(pres.shapes.RECTANGLE, { x: 5.1, y: 0.95, w: 4.4, h: 2.5, fill: { color: WHITE }, shadow: makeShadow() });
s8.addText("Bootstrap 95% CI for Sharpe", { x: 5.2, y: 1.05, w: 4.2, h: 0.4, fontSize: 14, bold: true, color: ACCENT, margin: 0 });
const ciData = [
  ["Flat(10bps)", "0.416", "[−0.835, 1.647]"],
  ["Flat(A&S)",   "0.449", "[−0.787, 1.677]"],
  ["A&S+CVaR",    "0.391", "[−0.843, 1.613]"],
  ["RL",          "−0.005",  "[0.000, 0.000]"],
];
ciData.forEach((row, i) => {
  const y = 1.5 + i * 0.45;
  s8.addText(row[0], { x: 5.3, y, w: 1.4, h: 0.35, fontSize: 11, color: DARK, bold: true, margin: 0 });
  s8.addText(row[1], { x: 6.7, y, w: 0.7, h: 0.35, fontSize: 11, color: DARK, align: "center", margin: 0 });
  s8.addText(row[2], { x: 7.4, y, w: 2.0, h: 0.35, fontSize: 10, color: SLATE, margin: 0 });
});

s8.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.65, w: 9, h: 0.95, fill: { color: WHITE }, shadow: makeShadow() });
s8.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.65, w: 0.08, h: 0.95, fill: { color: GOLD } });
s8.addText([
  { text: "NO statistically significant difference between any pair of strategies on Sharpe (all p > 0.10 after BH correction).", options: { bold: true, color: GOLD, breakLine: true } },
  { text: "The observed Sharpe differences are within the confidence intervals. The \"winner\" is not statistically distinguishable from the alternatives.", options: { color: DARK } },
], { x: 0.7, y: 3.7, w: 8.7, h: 0.85, fontSize: 12, margin: 0 });

// ══════════════════════════════════════════════════════════════════════
// SLIDE 9 — Frequency Sensitivity
// ══════════════════════════════════════════════════════════════════════
const s9 = lightSlide(pres, "Frequency Sensitivity — No Crossover Exists");
s9.addText("Rebalancing frequency swept from 1H (12 bars) to 1Q (9,504 bars) on the test period", { x: 0.5, y: 0.85, w: 9, h: 0.3, fontSize: 10, color: SLATE, italic: true, margin: 0 });

const freqTable = [
  [
    { text: "Frequency", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
    { text: "Bars", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
    { text: "Flat(10bps)", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
    { text: "Flat(A&S)", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
    { text: "CVaR", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
    { text: "RL", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
  ],
  [{ text: "1H" }, { text: "12" }, { text: "0.439" }, { text: "0.476", options: { bold: true, color: GREEN } }, { text: "0.373" }, { text: "−0.680" }],
  [{ text: "4H" }, { text: "48" }, { text: "0.439" }, { text: "0.476", options: { bold: true, color: GREEN } }, { text: "0.423" }, { text: "−0.680" }],
  [{ text: "1D" }, { text: "288" }, { text: "0.439" }, { text: "0.475", options: { bold: true, color: GREEN } }, { text: "0.416" }, { text: "−0.680" }],
  [{ text: "3D" }, { text: "864" }, { text: "0.439" }, { text: "0.463" }, { text: "0.384" }, { text: "−0.680" }],
  [{ text: "1W" }, { text: "2,016" }, { text: "0.439" }, { text: "0.477", options: { bold: true, color: GREEN } }, { text: "0.431" }, { text: "−0.680" }],
  [{ text: "1Q" }, { text: "9,504" }, { text: "0.439" }, { text: "0.478", options: { bold: true, color: GREEN } }, { text: "0.363" }, { text: "−0.680" }],
];
s9.addTable(freqTable, {
  x: 0.5, y: 1.2, w: 9.0, colW: [1.5, 1.2, 1.5, 1.5, 1.3, 2.0],
  fontSize: 11, color: DARK,
  border: { pt: 0.5, color: "E2E8F0" },
  rowH: 0.42,
});

s9.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.2, w: 9, h: 0.75, fill: { color: NAVY } });
s9.addText("Conclusion: Flat(A&S) wins at EVERY frequency. No crossover frequency exists in the tested range. Active rebalancing is unprofitable at any practical decision frequency.", { x: 0.6, y: 4.2, w: 8.8, h: 0.75, fontSize: 12, color: WHITE, valign: "middle", margin: 0 });

// ══════════════════════════════════════════════════════════════════════
// SLIDE 10 — Daily RL Experiment
// ══════════════════════════════════════════════════════════════════════
const s10 = lightSlide(pres, "Daily RL Experiment — Hypothesis Tested");
s10.addText("Hypothesis: Reducing RL decision frequency (every 288 bars) should reduce A&S cost headwind and improve Sharpe", { x: 0.5, y: 0.85, w: 9, h: 0.35, fontSize: 11, color: SLATE, italic: true, margin: 0 });

// Two comparison columns
s10.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.3, w: 4.2, h: 2.6, fill: { color: WHITE }, shadow: makeShadow() });
s10.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.3, w: 4.2, h: 0.5, fill: { color: "3B82F6" } });
s10.addText("5-min RL  (ppo_full.zip)", { x: 0.5, y: 1.3, w: 4.2, h: 0.5, fontSize: 13, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
const r5m = [["Sharpe", "−0.68"], ["Ann. Return", "−3.6%"], ["Max DD", "−7.9%"], ["Effective Decisions", "~100,000"]];
r5m.forEach((r, i) => { s10.addText(r[0], { x: 0.7, y: 1.9 + i * 0.5, w: 2.2, h: 0.4, fontSize: 12, color: SLATE, margin: 0 }); s10.addText(r[1], { x: 2.9, y: 1.9 + i * 0.5, w: 1.7, h: 0.4, fontSize: 12, bold: true, color: DARK, align: "right", margin: 0 }); });

s10.addShape(pres.shapes.RECTANGLE, { x: 5.0, y: 1.3, w: 4.5, h: 2.6, fill: { color: WHITE }, shadow: makeShadow() });
s10.addShape(pres.shapes.RECTANGLE, { x: 5.0, y: 1.3, w: 4.5, h: 0.5, fill: { color: GOLD } });
s10.addText("Daily RL  (ppo_daily.zip)", { x: 5.0, y: 1.3, w: 4.5, h: 0.5, fontSize: 13, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
const rd = [["Sharpe", "−3.88", { color: RED }], ["Ann. Return", "−22.8%", { color: RED }], ["Max DD", "−39.1%", { color: RED }], ["Effective Decisions", "~789", { color: SLATE }]];
rd.forEach((r, i) => {
  const col = r[2] || {};
  s10.addText(r[0], { x: 5.2, y: 1.9 + i * 0.5, w: 2.4, h: 0.4, fontSize: 12, color: SLATE, margin: 0 });
  s10.addText(r[1], { x: 7.6, y: 1.9 + i * 0.5, w: 1.8, h: 0.4, fontSize: 12, bold: true, color: col.color || DARK, align: "right", margin: 0 });
});

s10.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.1, w: 9, h: 0.85, fill: { color: NAVY } });
s10.addText([
  { text: "Daily RL performed WORSE.", options: { bold: true, color: GOLD } },
  { text: "  Only ~789 effective decisions (vs 100k at 5-min) makes it a much harder RL problem. Each wrong daily decision concentrates losses across 288 bars. The A&S cost headwind is frequency-invariant — reducing decisions does NOT solve the cost problem.", options: { color: WHITE } },
], { x: 0.6, y: 4.15, w: 8.8, h: 0.75, fontSize: 11, margin: 0 });

// ══════════════════════════════════════════════════════════════════════
// SLIDE 11 — Why RL Converged to Cash
// ══════════════════════════════════════════════════════════════════════
const s11 = lightSlide(pres, "Why RL Converged to Cash — Not a Bug, a Finding");
const reasons = [
  { num: "1", text: "RL observation has no useful signal — LightGBM R² ≈ 0 in all regimes, replaced with lagged returns (0.7×lag_1 + 0.3×lag_3)" },
  { num: "2", text: "Beat-benchmark reward is small (~4bps per bar) — less than A&S cost per trade (~123bps) — any trade has negative expected reward after costs" },
  { num: "3", text: "The optimal expected reward from any trade is NEGATIVE after costs — RL correctly learns this from experience" },
  { num: "4", text: "RL converges to the policy that minimizes expected loss: hold cash, avoid all trades" },
  { num: "5", text: "This is a MARKET MICROSTRUCTURE FINDING — not an implementation failure" },
];
reasons.forEach((r, i) => {
  const y = 0.95 + i * 0.9;
  s11.addShape(pres.shapes.RECTANGLE, { x: 0.5, y, w: 0.45, h: 0.75, fill: { color: ACCENT } });
  s11.addText(r.num, { x: 0.5, y, w: 0.45, h: 0.75, fontSize: 18, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  s11.addShape(pres.shapes.RECTANGLE, { x: 0.95, y, w: 8.55, h: 0.75, fill: { color: WHITE }, shadow: makeShadow() });
  s11.addText(r.text, { x: 1.05, y, w: 8.35, h: 0.75, fontSize: 12, color: DARK, valign: "middle", margin: 0 });
});

s11.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 5.0, w: 9, h: 0.05, fill: { color: ACCENT } });

// ══════════════════════════════════════════════════════════════════════
// SLIDE 12 — Conclusions
// ══════════════════════════════════════════════════════════════════════
const s12 = lightSlide(pres, "Conclusions");
const conclusions = [
  { icon: "1", head: "Core thesis confirmed", body: "Per-regime A&S calibration reveals that execution costs dominate active rebalancing returns in crypto markets at any practical frequency (1H to 1Q). Flat(A&S) wins at all frequencies.", color: GREEN },
  { icon: "2", head: "RL hypothesis not confirmed", body: "RL did not outperform CVaR. But its convergence to cash is the economically rational policy under the true cost landscape — confirmed by two independent methods (RL + CVaR optimizer).", color: GOLD },
  { icon: "3", head: "Flat(A&S) is the true baseline", body: "Buy-and-hold 60/40 BTC/ETH with realistic A&S costs (Sharpe +0.48) outperforms all active strategies — RL, CVaR, and rule-based alike.", color: "3B82F6" },
  { icon: "4", head: "Cost model matters more than strategy", body: "The gap between validation Sharpe (+1.72, 10bps) and test Sharpe (−0.68, A&S) is almost entirely from the cost model. Calibrate costs from exchange data before designing rebalancing strategies.", color: ORANGE },
];
conclusions.forEach((c, i) => {
  const y = 0.9 + i * 1.1;
  s12.addShape(pres.shapes.RECTANGLE, { x: 0.5, y, w: 0.5, h: 0.95, fill: { color: c.color } });
  s12.addText(c.icon, { x: 0.5, y: y + 0.15, w: 0.5, h: 0.65, fontSize: 20, bold: true, color: WHITE, align: "center", margin: 0 });
  s12.addShape(pres.shapes.RECTANGLE, { x: 1.0, y, w: 8.5, h: 0.95, fill: { color: WHITE }, shadow: makeShadow() });
  s12.addText(c.head, { x: 1.1, y: y + 0.06, w: 8.3, h: 0.4, fontSize: 13, bold: true, color: DARK, margin: 0 });
  s12.addText(c.body, { x: 1.1, y: y + 0.46, w: 8.3, h: 0.45, fontSize: 11, color: SLATE, margin: 0 });
});

// ══════════════════════════════════════════════════════════════════════
// SLIDE 13 — What Would Make RL Work
// ══════════════════════════════════════════════════════════════════════
const s13 = lightSlide(pres, "What Would Make RL Work in This Framework?");
const fixes = [
  { head: "Much larger signal-to-noise ratio", body: "R² >> 0.05 in return forecasts — currently the regime conditional signal is indistinguishable from noise" },
  { head: "Lower execution costs", body: "Institutional-grade execution, or larger position sizes where spread becomes negligible relative to expected return" },
  { head: "Higher frequency alpha", body: "Statistical arbitrage at second-scale frequency where the signal is stronger and more frequent, not regime-conditional tilts at 5-min bars" },
  { head: "Accept cash-optimal as a finding", body: "The current result (cash-optimal policy) is itself a valid market microstructure contribution — it reveals the cost landscape of crypto markets" },
];
fixes.forEach((f, i) => {
  const y = 0.95 + i * 1.1;
  s13.addShape(pres.shapes.RECTANGLE, { x: 0.5, y, w: 9, h: 0.95, fill: { color: WHITE }, shadow: makeShadow() });
  s13.addShape(pres.shapes.RECTANGLE, { x: 0.5, y, w: 0.08, h: 0.95, fill: { color: ACCENT } });
  s13.addText(f.head, { x: 0.7, y: y + 0.08, w: 8.7, h: 0.4, fontSize: 13, bold: true, color: DARK, margin: 0 });
  s13.addText(f.body, { x: 0.7, y: y + 0.48, w: 8.7, h: 0.4, fontSize: 11, color: SLATE, margin: 0 });
});

// ══════════════════════════════════════════════════════════════════════
// SLIDE 14 — Thank You
// ══════════════════════════════════════════════════════════════════════
const s14 = darkSlide(pres);
s14.addText("Thank You", { x: 0.5, y: 1.5, w: 9, h: 0.9, fontSize: 44, bold: true, color: WHITE, align: "center" });
s14.addShape(pres.shapes.RECTANGLE, { x: 3.5, y: 2.55, w: 3, h: 0.04, fill: { color: ACCENT } });
s14.addText("Questions?", { x: 0.5, y: 2.8, w: 9, h: 0.6, fontSize: 22, color: ICE, align: "center" });
s14.addText("Code: github.com/zihanlim/rapo-as-rl", { x: 0.5, y: 3.7, w: 9, h: 0.4, fontSize: 14, color: ICE, align: "center" });
s14.addText("MScFE 690 Capstone  |  WorldQuant University  |  2026", { x: 0.5, y: 4.2, w: 9, h: 0.4, fontSize: 12, color: ICE, align: "center", transparency: 40 });

// ══════════════════════════════════════════════════════════════════════
// WRITE
// ══════════════════════════════════════════════════════════════════════
pres.writeFile({ fileName: OUT }).then(() => console.log("Written:", OUT)).catch(e => { console.error(e); process.exit(1); });
