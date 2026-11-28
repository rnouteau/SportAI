# SportAI frontend

Web interface for browsing SportAI analyses: the list of analysed videos, then
for each one the overlay video, the player stats and the heatmaps.

Built with React 19, TypeScript, Vite, React Router, TanStack Query and
Tailwind CSS. It only reads data: everything comes from the FastAPI server in
`../api`.

## Run it locally

Requires Node.js (LTS). From this `frontend/` folder:

```bash
npm install        # once, and after every package.json change
```

Start the API from the repository root (in another terminal):

```bash
uvicorn api.main:app --reload
```

Then start the frontend:

```bash
npm run dev        # http://localhost:5173
```

In development, Vite forwards `/api` and `/files` to the API on
`http://127.0.0.1:8000` (see `vite.config.ts`), so the browser only talks to
Vite.

## Project structure

```
src/
├── main.tsx            # Entry point: providers (TanStack Query, router) around <App />
├── App.tsx             # The routes: / (list), /analyses/:id (detail), * (404)
├── api/
│   ├── client.ts       # fetch functions used by the pages (fetchAnalyses, fetchAnalysis)
│   └── types.ts        # Generated from the API's OpenAPI schema, do not edit
├── pages/              # One component per route; they load the data with useQuery
├── components/         # Display pieces receiving their data as props (cards, video, layout)
├── utils/format.ts     # Number / duration / file name formatting
└── test/               # Vitest setup + helpers (providers, fake fetch, sample analysis)
```

## API types

`src/api/types.ts` is generated from the API's OpenAPI schema. Regenerate it,
with the API running, whenever `api/schemas.py` changes:

```bash
npm run gen:types
```

## Tests

Vitest runs the tests in Node, with jsdom as a fake browser and Testing Library to
render the components and query them as a user would (by role, label, text):

```bash
npm test             # run once (what the CI does)
npm run test:watch   # re-run on every save
```

Test files sit next to the code they test (`format.ts` / `format.test.ts`). The page
tests replace `fetch` with a fake API (`src/test/render.tsx`): no server is needed.

The GitHub workflow `.github/workflows/frontend.yml` runs `npm ci`, `npm run lint`,
`npm test` and `npm run build` on every change of `frontend/`.

## Other scripts

| command | what it does |
|---|---|
| `npm run lint` | ESLint |
| `npm run build` | type check, then production build into `dist/` |
| `npm run preview` | serve the production build locally |
