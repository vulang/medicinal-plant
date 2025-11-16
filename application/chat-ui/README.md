# Medicinal Plant Chat UI

Angular-based chat interface that talks to the FastAPI classifier via `/predict`. Users upload an image inside a ChatGPT-like layout and receive the plant prediction as the assistant reply.

## Prerequisites
- Node.js 18+
- npm 9+

## Setup
```bash
cd application/chat-ui
npm install
```

## Development server
```bash
npm start
```
Runs `ng serve` on http://localhost:4200. The app expects the classifier API at `http://localhost:8000` (configure in `src/environments/environment.ts`).

## Build
```bash
npm run build
```
Outputs production files to `dist/chat-ui/`.

## Configuration
- `src/environments/environment.ts`: local API base URL.
- `src/environments/environment.prod.ts`: production API base URL. Update to match your deployed FastAPI endpoint.
