# deployment/ — Frontend

Next.js + D3.js interactive visualization, deployed on Vercel.

## How to Start

```bash
cd deployment

# Install deps if first time (required — concurrently isn't globally installed)
npm install

# Start Next.js directly
npx next dev --port 3000
```

**Do NOT use `npm run dev`** — the dev script calls `concurrently` (for ngrok tunneling) which isn't installed. Use `npx next dev` directly.

## Port Conflict Resolution

```bash
lsof -ti:3000 | xargs kill -9    # kill stale process
npx next dev --port 3000          # retry
```

## Component Map

| File | Purpose |
|------|---------|
| `app/page.tsx` | Main page — loads and displays graph |
| `app/layout.tsx` | App shell layout |
| `app/reference/page.tsx` | Reference documentation page |
| `components/GraphizVisualization.tsx` | Main D3.js graph renderer — zoom, pan, node rendering |
| `components/GraphizControls.tsx` | User interaction controls — filters, color-by property |
| `components/GraphizRolloutPanel.tsx` | Rollout detail sidebar — shows full CoT text |
| `components/GraphizLegend.tsx` | Property color legend |
| `components/GraphizReference.tsx` | Help/reference overlay |
| `components/AlgorithmStructure.tsx` | Algorithm-level graph view |
| `components/ClusterAlgorithmStructure.tsx` | Combined cluster + algorithm view |

## API Routes

| Route | Purpose |
|-------|---------|
| `app/api/files/route.ts` | List available flowcharts |
| `app/api/algorithms/[prompt]/route.ts` | Serve algorithm data per prompt |
| `app/api/rollout/[prompt]/[model]/[rid]/route.ts` | Serve individual rollout data |
| `app/api/flowchart/*/route.ts` | Serve static flowchart JSON (multiple routes) |

## Frontend Changes (Thought Anchors)

- **Node labels show anchor category** — when `anchor_category` is present on a node, the label displays `"{CATEGORY} {freq}"` (e.g., "AC 329") instead of just the frequency number. This is in `GraphizVisualization.tsx` line ~2187.
- Only applies to thought anchor flowcharts — regular flowcharts without `anchor_category` still show frequency only.
