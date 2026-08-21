# Dashboard Chat — how to ask

The **Ask for a chart** box on Stage 4 turns a plain-language request into a
chart or a written answer.

It is **rule-based first**: most requests are resolved instantly, offline, with
no API call and no free-tier quota spent. Only when the rules cannot work out
which columns you meant does it ask Gemini, and even then the answer is checked
against your real columns — so it can never chart a column that does not exist.

Works in **English and Spanish**.

---

## Chart or answer?

The chat decides which you wanted:

| You write | You get |
|---|---|
| `total sales by store` | a chart |
| `show me a chart of sales by store` | a chart |
| `what is the total sales` | a written answer, no chart |
| `how much did we sell` | a written answer, no chart |
| `total sales by store no graph` | a written answer, no chart |

The rule: saying **chart / graph / bar / pie / line** (or *gráfico, barras,
pastel, línea*) forces a chart. Saying **no graph** (or *sin gráfico*) forces
text. Otherwise, a **question** ("what is…", "how much…", *"cuánto…", "cuál
es…"*) is answered in words, and anything else draws a chart.

Charts also report the headline figures underneath, so you never have to read
values off an axis:

> Added: total **Weekly_Sales** by **Store**, as a **bar** chart.
> Total Weekly_Sales is **$114,436,974.62**; Store **16** leads with
> **$2,826,648.73** (2% of the total).

---

## What it can do

### Group by
- Text columns with a reasonable number of categories (`region`, `status`)
- **Numeric categories** such as `Store` (1–45) or `Holiday_Flag` (0/1)
- The **date column**, for anything over time

### Measure and aggregation
- **Sum** (default): `total sales by store`
- **Average**: `average temperature by store`, *`promedio de temperatura`*
- **Count of rows**: `store as pie chart`, or any request naming no measure

If you do not name a measure, it picks the table's main one (revenue / sales /
amount / total / price / …) when you ask for a ranking or talk about money;
otherwise it counts rows.

### Top N
`top 5 stores by sales`, `which stores make the most money`,
*`las tiendas que más venden`*

Accepts a **two-digit** number, so `top 5` through `top 99` work. `top 100` is
not recognised as a number and the request falls back to showing every
category. Saying "top" without a number defaults to **10**.

### Over time
`sales over time`, `weekly sales`, `monthly sales`, *`ventas semanales`*

Time requests draw a **line** and bucket the dates: *weekly* means weekly
totals, not one dot per calendar day. Granularities: **day, week, month,
quarter, year**. "Top N" is ignored on a timeline, because ranking a continuous
axis is meaningless.

### Chart types
**bar**, **line**, **pie** — and every chart has a dropdown next to its title
so you can switch it afterwards, including to **area**. The colour picker in
**Customize dashboard** applies to these charts too.

### Naming columns loosely
- Partial names: `sales` finds `Weekly_Sales`
- Business words: `stores` → `Store`, `sellers`, `products`, `customers`,
  `regions`, `categories`, `countries`, and the Spanish equivalents
  (*tiendas, vendedores, productos, clientes, zonas*)
- Near-spellings across languages: *`temperatura`* → `Temperature`

### Example requests that work
```
total sales by store
top 5 stores selling the most
which stores make the most money
average temperature by store
sales by holiday
weekly sales
monthly sales over time
store as pie chart
top 5 amount by region as pie chart
what is the total sales
what is the average temperature
ventas por tienda
las tiendas que más venden
quiero ver las ventas totales semanales
promedio de temperatura por tienda
cuánto es el total de ventas
```

---

## What it cannot do

### ⚠️ Silently ignored — the important list

These **do not fail**. They produce a chart that answers a *different*
question, so check the confirmation line, which always states exactly what was
charted.

| You write | What you actually get |
|---|---|
| `sales for store 5` | **all** stores — the filter is ignored |
| `sales only on holidays` | grouped **by** Holiday_Flag, not filtered to holidays |
| `sales by store and holiday` | grouped by **Store only** — the second column is dropped |
| `median sales by store` | **sum**, not median |
| `max sales by store` | **sum**, not max |
| `minimum temperature by store` | **sum**, not min |
| `bottom 5 stores by sales` | **all** stores — "bottom" is not understood |
| `percentage of sales by store` | plain **totals**, not percentages |
| `sales by store ascending` | sorted by **value, descending** |

**Rule of thumb: read the confirmation line.** It always spells out the
dimension, the measure, the aggregation and the top-N actually used.

### Refused clearly
These say so rather than guessing:

- **Filtering by date** — `sales in 2011`
- **Comparing periods** — `sales this year vs last year`
- **Scatter plots** — `scatter of temperature vs sales`
- **Histograms** — `histogram of sales`

### Not supported at all
- **Filtering / WHERE** of any kind (the biggest gap)
- **More than one grouping column** at a time
- Aggregations beyond **sum / average / count** — no median, min, max, count
  distinct, standard deviation
- **Calculations** — ratios, percentages, growth rates, running totals
- **Sort control** — always by value descending, or chronologically on a time
  axis
- Reading **other tables** or anything outside the current gold table
- **Changing your data** — the chat only reads; charts here never affect the
  approved dashboard, the plan or the export

---

## When it does not understand

You get the table's real columns grouped by role, plus runnable examples built
from **your** column names:

```
I'm not sure which columns you meant. Here's what this table has:

- Group by: Store, Holiday_Flag
- Time: Date
- Measure: Weekly_Sales, Temperature

Try one of these:
  total Weekly_Sales by Store
  top 5 Store by Weekly_Sales
  Weekly_Sales over time
  what is the total Weekly_Sales
```

If a column you expected is missing from that list, the cause is usually
**column roles**, not the chat:

- A text column with too many distinct values is not offered as a grouping
  column (an ID-like column would produce one bar per row).
- A numeric column counts as a **category** only when its values repeat a lot —
  that is what makes `Store` groupable while `Weekly_Sales` stays a measure.
- Columns named `*_id` are never treated as measures, because summing an
  identifier is meaningless.

---

## Tips

1. **Name the column** when unsure — exact names always win.
2. **Read the confirmation line.** It is the fastest way to catch a
   misunderstanding, especially given the silently-ignored list above.
3. **Say "no graph"** when you only want the number.
4. **Filter before the chat.** Filtering is not supported, so narrow the data
   in Stage 3 instead.
5. **Switch chart type afterwards** with the dropdown rather than rephrasing.
6. **Clear** removes every chat chart and starts over.

---

## Under the hood

- `crew/dashboard_agent/chat_intent.py` — request → `ChartRequest`
  (deterministic rules, optional LLM fallback). No Streamlit.
- `ui/dashboard_chat.py` — the chat panel and rendering; draws through the same
  aggregation the automatic dashboard uses.
- Model names come from `utils/config.py` (`DATACHEF_MODEL_CHAT`, then
  `DATACHEF_MODEL`, then a default) — see `.env.example`.

The LLM fallback runs on a model with a **separate free-tier quota** from the
transformation agent, so chatting can never consume the quota Stage 3 needs.
