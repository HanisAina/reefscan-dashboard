"""style.py: the look of the dashboard (clean, minimal product-page style).

All colours are defined once here. HTML helpers return markup for st.markdown.
"""
import html as _html
import dotmap

DOTS = dotmap.dotmap_uri()
DOTS_STRIP = dotmap.dotmap_uri(seed=11, lanes=6, cells=90, sand=0.5, path=False)

INK = "#111111"
MUTED = "#6e6e73"
SOFT = "#f5f5f7"
LINE = "#e5e5ea"
ACCENT = "#1259ff"
SAND = "#e9dcb5"
HEALTHY = "#3f9e6e"
BLEACHED = "#f2f2f2"
DANGER = "#e8384f"
MODE_COLORS = {"fast": "#c7c7cc", "normal": "#7aa7ff", "detail": "#e8384f"}
VEHICLE_COLORS = {"Fixed standard": "#8e8e93", "Fixed detailed": "#1c1c1e", "Adaptive": "#1259ff",
                  "Adaptive (no linger)": "#7aa7ff", "RL agent": "#00a383"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, .stMarkdown p, h1, h2, h3, h4, label, button, input, textarea,
[data-testid="stWidgetLabel"] p, [data-baseweb="select"] div {{
  font-family: 'Inter', 'Segoe UI', -apple-system, 'Helvetica Neue', Arial, sans-serif !important;
}}
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
.stApp {{ background: #ffffff; }}
.block-container {{ padding-top: 1.2rem !important; padding-bottom: 4rem !important; max-width: 1240px !important; }}
h1 {{ font-size: 3.1rem !important; font-weight: 800 !important; letter-spacing: -0.03em !important; line-height: 1.08 !important; color: {INK}; }}
h2 {{ font-size: 2.1rem !important; font-weight: 800 !important; letter-spacing: -0.02em !important; color: {INK}; }}
h3 {{ font-size: 1.25rem !important; font-weight: 700 !important; color: {INK}; }}
p, li {{ color: #333336; line-height: 1.6; }}

/* top navigation: radio buttons styled as pills */
div[role="radiogroup"] {{ gap: 6px !important; flex-wrap: wrap; }}
div[role="radiogroup"] > label {{
  background: {SOFT}; border-radius: 999px; padding: 7px 16px !important; margin: 0 !important;
  border: 1px solid transparent; transition: all .15s ease; cursor: pointer;
}}
div[role="radiogroup"] > label:hover {{ border-color: {INK}; }}
div[role="radiogroup"] > label > div:first-child {{ display: none !important; }}
div[role="radiogroup"] > label p {{ font-size: 0.92rem !important; font-weight: 600 !important; color: {INK} !important; }}
div[role="radiogroup"] > label:has(input:checked) {{ background: {INK}; }}
div[role="radiogroup"] > label:has(input:checked) p {{ color: #ffffff !important; }}

/* buttons */
.stButton > button, .stDownloadButton > button {{
  border-radius: 999px !important; background: {INK} !important; color: #fff !important; border: none !important;
  padding: 0.6rem 1.5rem !important; font-weight: 600 !important; transition: transform .1s ease, background .15s ease;
}}
.stButton > button:hover {{ background: #333 !important; transform: translateY(-1px); }}
[data-testid="stLinkButton"] a {{ border-radius: 999px !important; border: 2px solid {INK} !important; background: #fff !important;
  color: {INK} !important; font-weight: 600 !important; padding: 0.5rem 1.4rem !important; }}
[data-testid="stLinkButton"] a:hover {{ background: {SOFT} !important; }}
[data-testid="stLinkButton"] a p {{ color: {INK} !important; font-weight: 600 !important; }}
.stButton > button p {{ color: #fff !important; }}

/* expanders, tables, inputs */
[data-testid="stExpander"] {{ border: 1px solid {LINE} !important; border-radius: 18px !important; }}
[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; }}
[data-baseweb="select"] > div {{ border-radius: 12px !important; }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background: {INK} !important; border-color: {INK} !important; }}

/* custom blocks */
.brand {{ display:flex; align-items:center; justify-content:space-between; padding: 6px 2px 14px 2px; }}
.brand .logo {{ font-weight: 800; font-size: 1.15rem; letter-spacing: -0.02em; color: {INK}; }}
.brand .logo span {{ color: {ACCENT}; }}
.brand .meta {{ font-size: .82rem; color: {MUTED}; }}
.hero {{
  position: relative; overflow: hidden; border-radius: 28px; padding: 64px 56px 52px 56px; margin: 8px 0 28px 0;
  background: radial-gradient(1200px 500px at 85% -10%, #1f3fff55, transparent 60%),
              radial-gradient(800px 400px at 0% 120%, #00a38333, transparent 60%), #050507;
  color: #fff;
}}
.hero.art {{ min-height: 500px; padding: 60px 56px 52px 56px;
  background:
    url("__DOTS__") right 36px center / 48% auto no-repeat,
    radial-gradient(900px 420px at 80% 10%, #1d4ed855, transparent 60%),
    radial-gradient(700px 380px at 10% 110%, #0d948855, transparent 60%),
    linear-gradient(120deg, #03040a, #071233, #0a1c4a, #03040a);
  background-size: 48% auto, 100% 100%, 100% 100%, 300% 300%;
  animation: drift 18s ease-in-out infinite; }}
@keyframes drift {{ 0% {{ background-position: right 40px center, 0 0, 0 0, 0% 50%; }}
                   50% {{ background-position: right 40px center, 0 0, 0 0, 100% 50%; }}
                   100% {{ background-position: right 40px center, 0 0, 0 0, 0% 50%; }} }}
.hero .grad {{ background: linear-gradient(90deg, #7aa7ff, #2dd4bf); -webkit-background-clip: text; background-clip: text; color: transparent; }}
.hero .chips {{ display:flex; gap:10px; flex-wrap:wrap; margin-top: 26px; }}
.hero .chip {{ font-size:.82rem; color:#dbe4ff; background: rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.14);
  border-radius:999px; padding:6px 12px; backdrop-filter: blur(6px); }}
.hero .chip i {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; vertical-align:0; }}
.hero.art .text {{ max-width: 500px; }}
.hero.art p {{ max-width: 470px; }}
.hero.art h1 {{ font-size: 3.0rem !important; }}
.strip {{ height: 92px; border-radius: 20px; margin: 4px 0 18px 0;
  background: url("__STRIP__") right 24px center / auto 80% no-repeat,
              radial-gradient(700px 200px at 85% -20%, #1f3fff66, transparent 60%), linear-gradient(90deg, #03040a, #0a1c4a); }}
/* sticky frosted navigation */
.stMainBlockContainer div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stRadio"] div[role="radiogroup"][aria-label="Navigation"]) {{
  position: sticky; top: 0; z-index: 999; background: rgba(255,255,255,.82); backdrop-filter: saturate(180%) blur(14px);
  padding: 8px 0; border-bottom: 1px solid {LINE}; }}
/* motion */
.card, .feature, .robot {{ transition: transform .18s ease, box-shadow .18s ease; }}
.card:hover, .feature:hover, .robot:hover {{ transform: translateY(-3px); box-shadow: 0 14px 30px rgba(17,17,17,.08); }}
.card {{ border-top: 4px solid transparent; }}
.card.c0 {{ border-top-color: {ACCENT}; }} .card.c1 {{ border-top-color: #00a383; }}
.card.c2 {{ border-top-color: #ff9f0a; }} .card.c3 {{ border-top-color: #e8384f; }}
.feature .num {{ width:34px; height:34px; border-radius:12px; display:inline-flex; align-items:center; justify-content:center;
  font-weight:800; color:#fff; margin-bottom: 10px; }}
.hero .kicker {{ text-transform: uppercase; letter-spacing: .18em; font-size: .78rem; color: #9fb4ff; font-weight: 700; }}
.hero h1 {{ color: #fff !important; font-size: 3.5rem !important; margin: 12px 0 14px 0; max-width: 900px; }}
.hero p {{ color: #c7c7cc; font-size: 1.15rem; max-width: 760px; }}
.hero .stats {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 18px; margin-top: 40px; }}
.hero .stat .v {{ font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em; color:#fff; }}
.hero .stat .l {{ font-size: .88rem; color: #a1a1a6; line-height: 1.35; }}
.section {{ margin: 42px 0 14px 0; }}
.section .eyebrow {{ text-transform: uppercase; letter-spacing: .16em; font-size: .75rem; color: {ACCENT}; font-weight: 700; }}
.section h2 {{ margin: 6px 0 6px 0 !important; }}
.section p {{ color: {MUTED}; font-size: 1.05rem; max-width: 820px; margin: 0; }}
.cards {{ display:grid; gap: 16px; margin: 6px 0 10px 0; }}
.card {{ background: {SOFT}; border-radius: 22px; padding: 24px 24px 22px 24px; }}
.card .v {{ font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; color: {INK}; line-height: 1.1; }}
.card .l {{ font-size: .92rem; color: {INK}; font-weight: 600; margin-top: 6px; }}
.card .n {{ font-size: .82rem; color: {MUTED}; margin-top: 4px; }}
.card .n.up {{ color: #00875a; }} .card .n.down {{ color: {DANGER}; }}
.feature {{ background: {SOFT}; border-radius: 22px; padding: 26px; }}
.feature .tag {{ display:inline-block; font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
  color:{ACCENT}; background:#e8eeff; border-radius:999px; padding:4px 10px; }}
.feature h3 {{ margin: 12px 0 6px 0 !important; }}
.feature p {{ color: {MUTED}; font-size: .95rem; margin: 0; }}
.callout {{ border-radius: 18px; padding: 16px 20px; margin: 10px 0 14px 0; font-size: .97rem; }}
.callout.info {{ background: #eef3ff; color: #102a7a; }}
.callout.warn {{ background: #fff4e5; color: #7a4100; }}
.callout.bad {{ background: #fdecee; color: #8a1023; }}
.callout.good {{ background: #e8f7f0; color: #0b5e3d; }}
.legend {{ display:flex; gap:14px; flex-wrap:wrap; font-size:.85rem; color:{MUTED}; margin: 2px 0 6px 0; }}
.legend i {{ display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; border:1px solid #d1d1d6; }}
.explain {{ display:grid; grid-template-columns: 1fr 1fr; gap: 0; border-radius: 22px; overflow:hidden;
  border: 1px solid {LINE}; margin: 4px 0 22px 0; }}
.explain > div {{ padding: 20px 24px; }}
.explain .what {{ background: #ffffff; }}
.explain .notice {{ background: #f0f5ff; }}
.explain .h {{ font-size:.74rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; margin-bottom:6px; }}
.explain .what .h {{ color: {MUTED}; }} .explain .notice .h {{ color: {ACCENT}; }}
.explain p {{ margin: 0; font-size: .98rem; color: #26262a; }}
.explain ul {{ margin: 0; padding-left: 18px; }} .explain li {{ font-size: .96rem; color:#26262a; margin: 2px 0; }}
.step {{ display:inline-block; font-size:.75rem; font-weight:700; color:{MUTED}; background:{SOFT};
  border-radius:999px; padding:4px 12px; margin-bottom: 4px; }}
.robot {{ background:{SOFT}; border-radius:20px; padding:18px 18px 16px 18px; }}
.robot .dot {{ width:14px; height:14px; border-radius:50%; display:inline-block; margin-right:8px; vertical-align:-1px; }}
.robot b {{ font-size: 1rem; color:{INK}; }}
.robot p {{ font-size:.88rem; color:{MUTED}; margin:8px 0 0 0; line-height:1.45; }}
.tour {{ counter-reset: t; display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; }}
.tour div {{ background:{SOFT}; border-radius:18px; padding:16px 18px; font-size:.92rem; color:#26262a; }}
.tour div::before {{ counter-increment: t; content: counter(t); display:inline-flex; align-items:center; justify-content:center;
  width:26px; height:26px; border-radius:50%; background:{INK}; color:#fff; font-weight:700; font-size:.8rem; margin-right:10px; }}
@media (max-width: 900px) {{ .explain {{ grid-template-columns: 1fr; }} .tour {{ grid-template-columns: 1fr 1fr; }} }}
@media (max-width: 700px) {{
  .cards {{ grid-template-columns: repeat(2, minmax(0,1fr)) !important; }}
  .tour {{ grid-template-columns: 1fr !important; }}
  .promise td:first-child {{ width: auto; }}
  h1 {{ font-size: 2.2rem !important; }} h2 {{ font-size: 1.6rem !important; }}
  .hero.art {{ min-height: 0; padding: 32px 22px !important; }}
}}
.promise {{ width:100%; border-collapse: separate; border-spacing: 0 8px; }}
.promise td {{ background:{SOFT}; padding: 14px 16px; vertical-align: top; font-size: .93rem; color:#26262a; }}
.promise td:first-child {{ border-radius: 16px 0 0 16px; width: 30%; }}
.promise td:last-child {{ border-radius: 0 16px 16px 0; }}
.promise .k {{ font-size:.7rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; color:{MUTED}; display:block; margin-bottom:3px; }}
.promise th {{ text-align:left; font-size:.75rem; letter-spacing:.12em; text-transform:uppercase; color:{MUTED}; padding: 0 16px; }}
.pill {{ display:inline-block; border-radius:999px; padding:3px 10px; font-size:.78rem; font-weight:700; white-space:nowrap; }}
.pill.ok {{ background:#e8f7f0; color:#0b5e3d; }} .pill.part {{ background:#fff4e5; color:#7a4100; }}
.pill.no {{ background:#fdecee; color:#8a1023; }} .pill.found {{ background:#eef3ff; color:#102a7a; }}
.footer {{ margin-top: 60px; padding-top: 18px; border-top: 1px solid {LINE}; color: {MUTED}; font-size: .82rem; }}
@media (max-width: 900px) {{
  .hero.art {{ background-size: 0 0, 100% 100%, 100% 100%, 300% 300%; }}
  .hero.art .text {{ max-width: 100%; }}
  .hero {{ padding: 40px 26px; }} .hero h1 {{ font-size: 2.3rem !important; }}
  .hero .stats {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
}}
</style>
""".replace("__DOTS__", DOTS).replace("__STRIP__", DOTS_STRIP)


def esc(s):
    return _html.escape(str(s))


def brand():
    return ('<div class="brand"><div class="logo">Reef<span>Scan</span> · Survey Agent</div>'
            '<div class="meta">FYP 22005632 · Universiti Teknologi PETRONAS</div></div>')


def hero(kicker, title, text, accent="", chips=()):
    """Dark animated banner with the reef dot-map. `accent` is shown in gradient colour after the title."""
    acc = f' <span class="grad">{esc(accent)}</span>' if accent else ""
    ch = "".join(f'<span class="chip"><i style="background:{c}"></i>{esc(t)}</span>' for c, t in chips)
    return (f'<div class="hero art"><div class="text"><div class="kicker">{esc(kicker)}</div><h1>{esc(title)}{acc}</h1>'
            f'<p>{esc(text)}</p><div class="chips">{ch}</div></div></div>')


def strip():
    """Slim dark banner at the top of each page."""
    return '<div class="strip"></div>'


def section(eyebrow, title, text=""):
    p = f"<p>{esc(text)}</p>" if text else ""
    return f'<div class="section"><div class="eyebrow">{esc(eyebrow)}</div><h2>{esc(title)}</h2>{p}</div>'


def cards(items, cols=4):
    """items: (value, label, note, tone) with tone in '', 'up', 'down'."""
    out = []
    for it in items:
        v, l = it[0], it[1]
        n = it[2] if len(it) > 2 else ""
        tone = it[3] if len(it) > 3 else ""
        nn = f'<div class="n {tone}">{esc(n)}</div>' if n else ""
        out.append(f'<div class="card c{len(out) % 4}"><div class="v">{esc(v)}</div><div class="l">{esc(l)}</div>{nn}</div>')
    return f'<div class="cards" style="grid-template-columns: repeat({cols}, minmax(0,1fr));">{"".join(out)}</div>'


def features(items, cols=3):
    pal = [ACCENT, "#00a383", "#ff9f0a", "#e8384f"]
    out = [f'<div class="feature"><div class="num" style="background:{pal[i % 4]}">{i + 1}</div><br>'
           f'<span class="tag">{esc(t)}</span><h3>{esc(h)}</h3><p>{esc(p)}</p></div>'
           for i, (t, h, p) in enumerate(items)]
    return f'<div class="cards" style="grid-template-columns: repeat({cols}, minmax(0,1fr));">{"".join(out)}</div>'


def callout(text, tone="info"):
    return f'<div class="callout {tone}">{text}</div>'


def legend(items):
    return '<div class="legend">' + "".join(f'<span><i style="background:{c}"></i>{esc(l)}</span>' for c, l in items) + "</div>"


def explain(what, notice):
    """'What you are seeing' and 'What to notice' box. `notice` may be a list of bullet points."""
    if isinstance(notice, (list, tuple)):
        nb = "<ul>" + "".join(f"<li>{n}</li>" for n in notice) + "</ul>"
    else:
        nb = f"<p>{notice}</p>"
    return (f'<div class="explain"><div class="what"><div class="h">What you are seeing</div><p>{what}</p></div>'
            f'<div class="notice"><div class="h">What to notice</div>{nb}</div></div>')


def step(i, n):
    return f'<span class="step">Step {i} of {n}</span>'


def robots(items, cols=5):
    out = [f'<div class="robot"><span class="dot" style="background:{c}"></span><b>{esc(name)}</b><p>{esc(d)}</p></div>'
           for name, c, d in items]
    return f'<div class="cards" style="grid-template-columns: repeat({cols}, minmax(0,1fr));">{"".join(out)}</div>'


def tour(items):
    return '<div class="tour">' + "".join(f"<div>{t}</div>" for t in items) + "</div>"


def promise_table(rows):
    """rows: (kind, promise, status, evidence) with status in ok / part / no / found."""
    label = {"ok": "Achieved", "part": "Partly", "no": "Not done", "found": "Tested: new finding"}
    body = "".join(f'<tr><td><span class="k">{esc(k)}</span>{esc(p)}</td>'
                   f'<td style="width:130px"><span class="pill {st}">{label[st]}</span></td><td>{ev}</td></tr>'
                   for k, p, st, ev in rows)
    return ('<table class="promise"><tr><th>Interim report promised</th><th>Status</th><th>What the evidence shows</th></tr>'
            + body + "</table>")


def footer():
    return ('<div class="footer">Autonomous Resource-Aware Agent for Priority-Based Coral Reef Health Mapping · '
            'Muhammad Alif Najmi bin Mohammad Zamri · Supervisor: Ts. Dr. Manoranjitham A/P Muniandy · '
            'Co-supervisor: Ts. Dr. Mazeyanti Bt Mohd Ariffin · Simulation-only study</div>')
