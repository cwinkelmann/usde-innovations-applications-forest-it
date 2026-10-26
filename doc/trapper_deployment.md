# Running Trapper alongside the USDE webapp

Battle-tested on the ThinkPad T14 (Ubuntu, Docker 27.1.1, Compose v2.29.1) on 2026-10-17 from
`~/work/trapper-setup` (commit `ebdd5c4`, synced to `t14:~/work/trapper-setup-test`).
**Nothing has been run on carrot yet** — carrot is the final deployment target; this records the
procedure and every deviation from the official docs that was needed.

## Why "localhost mode"

Trapper's production mode wants Traefik on ports 80/443 plus DNS + SSL. On a shared server that
already hosts other services, use **localhost mode** instead (`docker-compose.localhost.yml`):
Traefik is disabled (`replicas: 0`) and every service is exposed on its own remappable host port.
The stack lives in its own compose project + network, so it coexists with the USDE webapp compose
(ports 8099/8101 on carrot) without interaction.

## Setup procedure (as executed on t14)

```bash
# 1. Get the repo onto the host
rsync -a --exclude 'profiles/' ~/work/trapper-setup/ <host>:~/work/trapper-setup-test/

# 2. Generate a localhost profile — NON-INTERACTIVE needs the GPU answer piped (see deviations)
cd ~/work/trapper-setup-test
printf 'n\n' | uv run ./configure.py localhost \
    --modules trapper,trapper-ai,ai-worker --accelerator cpu
# → writes profiles/localhost-dev/.env (secrets auto-generated) + start/stop/logs/ps/exec helpers

# 3. Remap colliding host ports in profiles/localhost-dev/.env
#    (8080/8081 were taken by CVAT on t14; on carrot 8080=charm-api, 8001 and 5555 are taken too)
TRAPPER_NGINX_DEV_PORT=8102     # main Trapper Expert UI (was 8080)
TRAPPER_NGINX_AI_DEV_PORT=8103  # Trapper AI nginx      (was 8081)
FLOWER_DEV_PORT=8104            # Celery monitoring     (was 5555)
TRAPPER_AI_DEV_PORT=8105        # Trapper AI Django     (was 8001)
# kept: 8000 (trapper), 8088 (uploader), 5432/5433 (postgres), 15672 (rabbitmq UI)

# 4. Start with PRE-BUILT images — bypass the generated start.sh (see deviations)
COMPOSE_DISABLE_ENV_FILE=1 docker compose \
    -f docker-compose.yml -f docker-compose.localhost.yml \
    --env-file profiles/localhost-dev/.env \
    --profile trapper --profile trapper-ai --profile ai-worker up -d

# 5. One-time initialisation, non-interactive (superusers, AI-worker user, taxonomy,
#    prediction models, classificators, test projects)
uv run ./admin.py quick-setup -p localhost-dev
```

Result on t14: 15 containers up, Django migrations applied, HTTP 200 on 8000/8102/8103/8105,
AI worker connected to RabbitMQ after wizard step 3 created its API user
(`ACCESS_REFUSED` for user `trapperai` before that step is **expected**, not a bug).
**UI login test passed** (2026-10-17): logged in as `admin` at `http://<host>:8102`, dashboard
renders with Messages / Research / Classification / Data packages / Celery+Remote task panels —
but only after disabling the debug toolbar (deviation 8 below).

## Deviations from the official README

1. **Never use the system Python.** The README suggests `pip install typer rich`;
   `configure.py` even tries to self-install via `sys.executable -m pip`, which crashes on hosts
   whose python has no pip (t14). The repo ships `pyproject.toml` + `uv.lock` — run everything as
   `uv run ./configure.py …` / `uv run ./admin.py …`.
2. **`configure.py localhost` is not actually non-interactive when a GPU is present.** Even with
   `--accelerator cpu` it prompts "Enable GPU support for AI Worker? [y/n]" and aborts without a
   TTY. Workaround: `printf 'n\n' |` pipe. (Upstream bug candidate.)
3. **The generated `start.sh` is dev-mode only.** It stacks `docker-compose.dev.yml`, which
   bind-mounts source from `TRAPPER_CODE_DIR`/`TRAPPER_AI_CODE_DIR` (`~/work/trapper`,
   `~/work/trapper-ai`). Without those checkouts the containers would start empty. For a
   course/production-style deployment, compose manually with only base + localhost overlays
   (step 4 above) so the pulled registry images are used as-is.
4. **Stale `docker login` breaks the pull.** A leftover credential for `registry.gitlab.com`
   caused `unauthorized: HTTP Basic: Access denied` on every `trapper-project` image. The registry
   is public — `docker logout registry.gitlab.com` fixes it. Beware: `docker compose up -d` still
   **exited 0** despite the failed pull, so verify with `docker compose ps`.
5. **`ALLOWED_HOSTS` only gets one auto-detected IP.** configure.py wrote `192.168.188.166` but
   t14 also answers on `192.168.188.88` — requests via the second IP get Django 400s. On carrot,
   append `10.188.1.1` (and any hostname) to `TRAPPER_ALLOWED_HOSTS` and
   `TRAPPER_AI_ALLOWED_HOSTS` in the profile `.env`.
6. **Used `admin.py quick-setup`** (undocumented in the README, which only shows the interactive
   `wizard`) — runs the init steps without prompts, **but silently skips superuser creation**
   ("requires interactive input"). Create the admins separately:
   ```bash
   docker exec -e DJANGO_SUPERUSER_USERNAME=admin \
     -e DJANGO_SUPERUSER_EMAIL=admin@os-conservation.org \
     -e DJANGO_SUPERUSER_PASSWORD="$PW" \
     trapper-localhost-dev-trapper-1 python manage.py createsuperuser --noinput
   # same for trapper-localhost-dev-trapper-ai-web-1
   ```
   (On t14 the password is stored as `TRAPPER_SUPERUSER_PASSWORD` in the profile `.env`.)
   `wizard --steps 5-8` is NOT a non-interactive alternative: with `--steps` the per-step
   confirmations are skipped, but step 5 still prompts for the provider selection.
7. **Model downloads hang forever if the disk fills.** Step 5 (`load_prediction_models`)
   downloads ~0.5–2 GB per model into the Trapper AI media volume with no socket timeout: when
   t14's disk ran full mid-download, the step hung indefinitely (0% CPU, frozen net counters,
   empty `ai_worker_model_cache`). Recovery: free space, kill the wizard + its `docker compose
   exec` child, `docker restart trapper-localhost-dev-trapper-ai-web-1`, re-run `quick-setup`
   (steps 1–4 are fast, ~7 min). Budget ≥20 GB free before step 5.
8. **`DEBUG=True` in the generated localhost profile breaks the UI with the prod images.** The
   profile sets `TRAPPER_DEBUG=True` + `USE_DEBUG_TOOLBAR=True`, but the pre-built image doesn't
   mount django-debug-toolbar's URLs → every logged-in page 500s with
   `NoReverseMatch: 'djdt' is not a registered namespace`. Set `TRAPPER_DEBUG=False`,
   `USE_DEBUG_TOOLBAR=False`, `TRAPPER_AI_DEBUG=False` in the profile `.env` and
   `docker compose … up -d` to recreate.
9. **`wizard --steps 5-8` cannot be scripted via piped stdin**: an internal
   `docker compose exec` (fetching the provider list) swallows the piped input, so the provider
   prompt gets EOF and the wizard aborts. Run the underlying commands directly instead:
   ```bash
   # 5a — in trapper-ai-web           (pick model codes; DeepFaune skipped here, see below)
   manage.py load_prediction_models --model-codes=md5_5a,md6_YOLOv9c,...,tai_02.2024
   # 5b — in trapper                  (then wizard --steps 6-8 for the rest)
   # NB: the current :latest image only accepts the LEGACY positional syntax —
   # the --trapper-url/--api-url flag syntax that admin.py tries first errors out.
   manage.py load_aiproviders http://172.17.0.1:8102 http://172.17.0.1:8103 \
     trapperai "$AI_WORKER_PASSWORD" --model-codes=...
   ```
   Run step 6 (classificators) only AFTER 5b has succeeded — it builds classificators from the
   providers registered in Expert and reports "Done" even when it created none.
10. **Variable-name bug in `admin.py` step 5b with remapped ports:** it reads
    `TRAPPER_AI_NGINX_DEV_PORT`, but the compose/.env variable is spelled
    `TRAPPER_NGINX_AI_DEV_PORT` — with remapped ports 5b silently targets the default 8081.
    Pass `--api-url` explicitly (as above). Works upstream only because the default port
    happens to match. (Upstream bug candidate.)
11. **DeepFaune downloads are throttled to ~60 KiB/s** by their host
    (`pbil.univ-lyon1.fr`, 134.214.0.0/16) — ~2–3 GB of weights = many hours, while MegaDetector
    weights come from GitHub at MB/s. For a first deployment, load only the MegaDetector +
    TrapperAI providers (deviation 9) and add DeepFaune later/overnight with a single
    `load_prediction_models --model-codes=dfd_1.1,dfc_1.2,dfc_1.3` + re-run of 5b/6.

## Relation to the official readthedocs installation page

`https://trapper-project.readthedocs.io/en/latest/installation.html` (docs v1.7.2) describes a
**different, older setup**: cloning the standalone `trapper` repo, `cp trapper.env .env`, and
`./start.sh -pb prod|dev`. That stack is Trapper Expert only — no Traefik, no Trapper AI, no AI
worker, and it builds images locally instead of pulling them. We deliberately use the newer
`trapper-setup` "Unified Docker Ecosystem" (its README is the authoritative doc) because the
course needs the AI pipeline (MegaDetector/DeepFaune classification via the AI worker).

## Carrot-specific notes (when we deploy for real)

- Port plan is already carrot-clean: 8000, 8088, 8102–8105 were free on carrot as of 2026-10-16
  (checked live; 8080, 8001, 5555, 8099–8101 are taken).
- Keep `GPU_ENABLED=false` initially. If the AI worker should use a GPU: `GPU_COUNT=1` — the
  default `all` would claim all 8 H100s — and book it via `hrm` like the webapp does.
- The localhost overlay binds `0.0.0.0`, so Postgres/RabbitMQ/Memcached become LAN-reachable.
  Students need the web UIs only; consider prefixing infra ports with `127.0.0.1:` in a copy of
  the overlay.
- Observed cosmetic issues on t14: celery "substantial drift" warnings (7200 s = container TZ
  mismatch, UTC vs CEST) and `Fontconfig error: No writable cache directories` in the AI worker.
  Neither blocked operation.
