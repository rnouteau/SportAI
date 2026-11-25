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

## API types

`src/api/types.ts` is generated from the API's OpenAPI schema. Regenerate it,
with the API running, whenever `api/schemas.py` changes:

```bash
npm run gen:types
```

## Other scripts

| command | what it does |
|---|---|
| `npm run lint` | ESLint |
| `npm run build` | type check, then production build into `dist/` |
| `npm run preview` | serve the production build locally |
