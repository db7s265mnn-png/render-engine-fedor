# Agent notes (Bob Render)

## ALWAYS give the user a Windows build link

After every push that produces a Windows binary, the user wants a **clickable GitHub Actions URL** to the build/artifact — not only a local `/opt/cursor/artifacts/...` path.

Example:
https://github.com/db7s265mnn-png/render-engine-fedor/actions/runs/<run-id>

Include the artifact name (e.g. `Bob-Render-windows-x64-0.9.0-<sha>`). Prefer waiting for CI and posting the run link before the final summary of that turn.
