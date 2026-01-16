# Claude Code Instructions

## FORBIDDEN - Never Do These
- **NEVER** run `source venv/bin/activate` or manually activate virtual environments
- **NEVER** run `./venv/bin/python` directly
- **NEVER** run `./venv/bin/pip` directly

## REQUIRED - Always Use npm Scripts
All commands go through npm. The venv is an implementation detail.

```bash
npm run setup    # First time setup (creates venv, installs deps)
npm test         # Run all tests
npm test:fast    # Run unit tests only (skip slow integration)
npm start        # Build and run the app
npm run dev      # Development mode with hot reload
```

## Why
- Anyone cloning this repo uses npm, not Claude Code
- Full automation - no manual environment setup
- Tests must pass via `npm test` before any commit
- If `npm test` fails, the code is broken - fix it
