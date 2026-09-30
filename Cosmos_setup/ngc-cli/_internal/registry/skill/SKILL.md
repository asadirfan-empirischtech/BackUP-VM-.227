---
name: registry
description: Reference for NGC CLI registry commands. Covers container images, models, resources, Helm charts, collections, CSPs, encryption keys, label-sets, and registry usage. Use when working with ngc registry, NGC catalog, container images, models, resources, charts, collections, CSP deployment, encryption keys, or registry storage.
compatibility: Requires NGC CLI installed and configured
metadata:
  author: "NGC CLI Team"
  version: "1.0.3"
  tags:
    - ngc
    - registry
    - container
    - model
    - resource
    - helm
    - collection
    - csp
    - cli
  domain: ngc-registry
---

# NGC Registry CLI Skill

Full catalog for `ngc registry` commands. For each artifact type, see the reference for that artifact in **References** below.

## How to use this skill

1. Identify the **artifact** (image, model, resource, chart, collection, csp, encryption-key, label-set) or **usage** (storage report only).
2. Open the reference for that artifact: [references/image.md](references/image.md), [references/model.md](references/model.md), etc.
3. In that reference, find the **action** (list, info, create, push, pull, upload-version, download-version, remove, or artifact-only action).
4. Run the command with `--format_type json` when it returns output.

This SKILL is the index and defines global rules; each reference gives that artifact's actions and exact commands.

## Prerequisites

- **NGC CLI** installed and on the PATH (`ngc` available).
- For **private or write operations:** configured authentication (e.g. API key) and correct org/team. Run `ngc config current` to verify; see **Before you start** and **Authentication and access** for details.

## Executable availability gate

Before running any `ngc` command, check whether `ngc` is available on PATH with `command -v ngc` or an equivalent direct PATH lookup. If `ngc` is not available, stop and tell the user to install or activate the NGC CLI, then retry after `ngc` is on PATH.

Do **not** search the filesystem for alternate `ngc` binaries, virtualenvs, project checkouts, or sibling installs (`find / -name ngc`, `ls ~/ngc-cli`, `.venv/bin/ngc`, `poetry run which ngc`, etc.). The registry skill requires the user's active shell environment to provide `ngc`.

## References (by artifact)

One reference per artifact; each spells out that artifact's actions and commands. Each reference is split into Read operations and Write operations.

image — [references/image.md](references/image.md)
model — [references/model.md](references/model.md)
resource — [references/resource.md](references/resource.md)
chart — [references/chart.md](references/chart.md)
collection — [references/collection.md](references/collection.md)
label-set — [references/label-set.md](references/label-set.md)
encryption-key — [references/encryption-key.md](references/encryption-key.md)
csp — [references/csp.md](references/csp.md)
usage — [references/usage.md](references/usage.md) (storage report only)

---

## Warnings (do not misunderstand)

- **usage** is **not** an artifact. Only `ngc registry usage info` exists. Never suggest list/create/update/remove/upload/download for usage.
- **image** version is called **tag**, not "version". Target uses `:tag`. Always say "tag" in instructions.
- **collection** and **label-set** have **no versions**. Do not suggest upload or download for them. Do not reference "collection version" or "label-set version".
- **encryption-key** has no versions and no create in this CLI (keys created elsewhere). Only list, info, remove, disassociate, status.
- For **"my"** or private artifacts, the user must be logged in. Suggest `ngc config current` before running commands.
- **publish is not allowed through this skill.** Do not run, suggest, or construct `ngc registry <artifact> publish` commands. If the user asks to publish, explain that publishing must be handled outside the agent skill workflow by an authorized human/process.
- **Do not substitute another write action.** If the user asks to create an artifact and the artifact already exists, do not update it. Report that it already exists and ask the user to explicitly request an update if that is what they want. Similarly, do not turn "remove versions" into "remove artifact".

## Write operations — sequential only

**Rule:** For **all artifacts**, do **not** run write operations in parallel. Run them **one after another** (sequentially). Run one command at a time; do not start the next until the previous has completed (e.g. exit code 0).

Write operations include: create, update, push, upload-version, remove, sign, set-latest, accept-license, update-license-terms, deploy start/update/remove, and any other action that modifies registry or artifact state. Read-only operations (list, info, usage info) may be run in parallel if needed.

## Write operations — preflight gate

Before any write operation (create, update, push, upload-version, remove, sign, set-latest, accept-license, update-license-terms, deploy start/update/remove), run `ngc config current`. If it shows no configured account, **MUST NOT continue** — tell the user to run `ngc config set` and enter their API key and default org/team, then wait for them to retry.

Do **not** run the write command "anyway to see the exact error" — this wastes the user's time and tokens and produces a misleading error instead of a useful next step.

`ngc config current` is the **only** source of NGC CLI configuration state. Do **not** try to fill in missing org/team/apikey by reading sibling project configs, account/credentials files (e.g. `account_*.config`), env files, or shell history. If `ngc config current` shows no configured account, the user must set the config themselves in their terminal (per **Do Not Modify NGC Configuration**) — do not bridge the gap by guessing values from another project on the host.

## Access-controlled read preflight

Before private or access-controlled read operations, including all `ngc registry encryption-key` commands, run `ngc config current`. If it shows no configured account, **MUST NOT continue** — tell the user to run `ngc config set` in their own terminal and wait for them to retry.

`ngc registry encryption-key` commands do not support cross-org lookup with `--org`; they use the currently configured org and optional `--list-team` scope. Do not try to work around missing config by retrieving credentials or API keys from another system.

## Output format — always JSON

**Rule:** For every `ngc registry` command you execute, always pass `--format_type json`. Do not rely on the user's default configuration. Other formats (ascii, csv) produce segmented or tabular output that is harder to parse reliably for agents.

```bash
ngc registry image list --format_type json
ngc registry model info org/my_model:1.0 --format_type json
```

Always append `--format_type json` to registry commands that support it. If the user asks for ASCII, CSV, a table, a numbered list, or another presentation format, still run the CLI with `--format_type json`, parse the JSON, then convert the response you show to the user's requested format. Only omit JSON when the CLI action does not accept `--format_type` (for example some transfer commands), or when the user explicitly asks you to execute the CLI with a specific raw `--format_type` for a formatting diagnostic/reproduction and you will not need to parse that output for follow-up decisions.

Do not run `ngc registry ... --format_type ascii` or `--format_type csv` just because the user asked for ASCII or CSV output in chat.

## Intent boundaries for writes

Write actions are not interchangeable. Match the user's requested verb and object scope exactly:

- If the user asks to **create** an artifact and it already exists, stop after reporting that it exists. Do not run `update` unless the user explicitly asks to update the existing artifact.
- If the user asks to **update** metadata, use `update`; do not create a replacement artifact.
- If the user asks to remove **versions/tags**, remove only version/tag targets and keep the top-level artifact metadata.
- If the user asks to remove the **artifact**, removing the top-level target is allowed after the normal write preflight and confirmation.

## Removing versions without removing artifact metadata

Versioned artifacts have top-level metadata plus versions underneath. Removing the top-level artifact target deletes the artifact metadata and may delete all versions; do this only when the user explicitly asks to remove/delete the artifact itself.

When the user asks to remove all versions, matching versions, old versions, tags, or similar while keeping the artifact:

- **image:** image tags support glob removal for a concrete repository. Use a tag pattern such as `ngc registry image remove <org>/[team/]<image>:* --format_type json` when the user confirms the matched tag scope.
- **model/resource/chart:** `remove` does not support version globs. First list versions with a concrete artifact plus version pattern, for example `ngc registry model list <org>/[team/]<model>:* --format_type json`. Parse the JSON version list, apply the user's requested version filter, show the exact version targets, get confirmation for those exact targets, then run one `remove <artifact>:<version> --format_type json` command at a time.

Do not run `ngc registry model remove <org>/[team/]<model>`, `ngc registry resource remove <org>/[team/]<resource>`, or `ngc registry chart remove <org>/[team/]<chart>` for a request that only says to remove versions.

## Before you start

```bash
ngc config current
```

Use `--org` and `--team` only to override command context; use `--team no-team` for org-level context. These flags are not artifact ownership filters for registry list results. For access-controlled artifacts, ensure the user is logged in and the active org/team matches. **If any registry command fails with an access-related error, run `ngc config current` first** to confirm login and org/team before doing anything else.

## Authentication and access

See **Before you start** for `ngc config current` and org/team overrides.

- **Guest mode:** Public catalog lookup without login (read-only).
- **Access-controlled artifacts:** For private, access-controlled resources or any write operations, the user **must be logged in**. List, info on private resources require authentication, create, update, upload, download, and remove for all artifacts require authentication.
- **API keys are org-scoped.** The active key determines which organization (and optionally team) the user is acting in. An org-scoped key can also access **subscribed artifacts across orgs** (artifacts shared or subscribed to by that org).
- **Check logged-in access:** Run `ngc config current` to check if user is logged in. Confirm org/team before running registry commands.

**When you get an access error** (403, 401, forbidden, "not a catalog artifact", authentication errors): Run `ngc config current` (see **Before you start**). If the user is not logged in or org/team is wrong, have them log in or fix config before retrying. In most access-related failures, checking config first is the right step.

**Registry connectivity diagnostics:** `ngc diag server --debug` checks the container registry through the Docker Registry `/v2/` endpoint and follows the advertised bearer-token endpoint. In guest mode, a trusted `/v2/` bearer challenge means the registry is reachable even though private image access still requires configured credentials.

## Do Not Modify NGC Configuration

**Never** change the active organization, team, API key, or any other NGC CLI configuration on behalf of the user. This includes `ngc config set`, `ngc config set --org`, `ngc config set --team`, `ngc config set --apikey`, `ngc config clear`, and any operation that writes to `~/.ngc/config`. Only the user may change their NGC configuration, and they must do so themselves in their terminal.

`ngc config` writes are **persistent and global**: they affect every future `ngc` command on the host (across shells, scripts, and tools — not just the current Claude Code session). Even if the user asks in chat ("change org to X", "switch team to Y", "set my api key"), do not run these commands. Ask the user to run them in their own terminal.

## Do Not Retrieve Credentials

**Never** retrieve, search for, offer to retrieve, or suggest retrieving API keys, credentials, tokens, or secrets from NGC Vault, password managers, secret stores, credential services, local files, shell history, project configs, or other systems. This applies even when the user gives an org ID or asks for help preparing credentials.

If a command requires authentication and `ngc config current` shows no configured account, stop after explaining that the user must obtain their own authorized API key and run `ngc config set` in their own terminal. Do not ask the user to paste API keys into chat.

If the user wants to run a command in a different org or team context than the one configured, use the per-command `--org` / `--team` flags where the command supports them — these scope a single command without mutating any state:

```bash
ngc registry <subcommand> <action> ... --org <org-name> --team <team-name> --format_type json
```

To discover available orgs/teams without modifying state:

```bash
ngc org list --format_type json
ngc team list --format_type json
```

The `--org` / `--team` flags accept the `name` field (not `displayName`).

For **registry list commands**, do not use `--team` as a team-owned artifact filter. To list artifacts under a specific team namespace, pass a quoted target pattern instead. Quote wildcard targets so the user's shell does not expand `*` before the CLI sees it:

```bash
ngc registry model list "<org-name>/<team-name>/*" --format_type json
ngc registry collection list "<org-name>/<team-name>/*" --format_type json
```

## Command structure

```bash
ngc registry <subcommand> [action] [options] [target] --format_type json
```

**Subcommand** = artifact type or report: `image`, `model`, `resource`, `chart`, `collection`, `label-set`, `encryption-key`, `csp`, or `usage`. (**usage** is a report only — only `usage info` exists; it is not an artifact.)

## Command tree at a glance

Artifact order matches **References** above. Upload/download column names match the CLI.

artifact — upload action — download action — versions
image — push — pull — Yes (tag)
model — upload-version — download-version — Yes
resource — upload-version — download-version — Yes
chart — push — pull — Yes
collection — (none) — (none) — No
label-set — (none) — (none) — No
encryption-key — (none) — (none) — No
csp — (none) — (none) — N/A
usage — (none) — (none) — Not an artifact; usage info only

Quick one-liners (add `--format_type json` when the command returns output; see references for resource, chart, label-set, encryption-key, csp):

```bash
ngc registry image list --format_type json
ngc registry model list --format_type json
ngc registry image push <org>/[team/]image[:tag]
ngc registry image pull <org>/[team/]image[:tag]
ngc registry model upload-version <org>/[team/]model:version ...
ngc registry model download-version <org>/[team/]model:version --format_type json
ngc registry collection list --format_type json
ngc registry usage info --format_type json
```

## Target formats

Same artifact order as References. "versioned" = has versions/tags in the CLI.

artifact — target format — versioned
image — org/[team/]image[:tag] — Yes (tag)
model — org/[team/]name or org/[team/]name:version — Yes
resource — org/[team/]name or org/[team/]name:version — Yes
chart — org/[team/]chart_name[:version] — Yes
collection — org/[team/]name — No
label-set — org/[team/]name — No
encryption-key — [org/[team/]]keyid — No
csp — CSP name — N/A
usage — (none) — Not an artifact

## Artifact naming

Artifacts are named by **org** and optionally **team**:

- **Org + team:** `<org>/<team>/<artifact-name>`
  Example: `nvidia/team-alpha/my-model`
- **Org, no team:** `<org>/<artifact-name>`
  Example: `nvidia/my-model`

**`<org>` and `<team>` mean the canonical `name` field, not the `displayName`.** Target paths and the `--org` / `--team` flags both expect `name`. Using `displayName` typically produces a 403 from the API.

`ngc config current` prints the org row in the format `<displayName> (<name>)` — e.g. `Example Org (example-org)` means `displayName = Example Org`, `name = example-org`. Use the parenthesized `name` (`example-org`) in target paths and flags. The same parsing applies to the team row when present. The `ngc config current --format_type json` output does **not** split these into separate fields — you still have to parse the value string.

When unsure, resolve names with `ngc org list --format_type json` / `ngc team list --format_type json` and use the `.name` field.

For versioned artifacts (image, model, resource, chart), append a version or tag: e.g. `<org>/<team>/<artifact-name>:<version>` or `<org>/<artifact-name>:<tag>`. Image versions are called **tags** (use `:tag` not `:version` for images).

## Presenting list and info results

When showing `list` or `info` output to the user, render each artifact as its **fully-qualified target path** — `<org>/[<team>/]<artifact-name>[:<version>]`, using the canonical `name` field for `<org>` / `<team>` (see **Artifact naming**) — not just the bare artifact name. This is the form the user can copy-paste directly into the next command (`info`, `download-version`, `pull`, `update`, `remove`, `push`, etc.).

Example for a model listing — instead of:

| Name | Public | Updated |
| --- | --- | --- |
| my-model | False | 2026-04-27 |

show:

| Target | Public | Updated |
| --- | --- | --- |
| example-org/my-model | False | 2026-04-27 |

If `--org` / `--team` were not explicitly passed, read them from `ngc config current` (apply the `<displayName> (<name>)` parse rule from **Artifact naming**). Include `:tag` for images and `:version` for model/resource/chart whenever versions are listed.

## Common options

--format_type json — Always use for agent-executed commands
--org <name> — Organization context (default from config)
--team <name> — Team context (no-team for org-level); not a list ownership filter
--ace <name> — ACE name (no-ace to override)
--debug — Debug mode

## Environment configuration

Check CLI version with `ngc --version`. Use the user's configured environment and do not add environment overrides unless the user explicitly provides them in their own terminal session.

## General actions (summary)

Read: list — List artifacts and/or versions (scope by pattern). info — Get metadata for artifact or version. download — Transfer to local: pull (image/chart), download-version (model/resource).

Write: create — Create artifact or version metadata. update — Update artifact or version metadata. upload — Populate version/tag: push (image/chart), upload-version (model/resource). accept-license — Display artifact license terms and accept required terms after confirmation. remove — Delete version or artifact.

Exact commands and artifact-specific actions are in each artifact's reference.

## Discovering commands

When intent is unclear or you need exact options:

```bash
ngc registry --help
ngc registry <subcommand> --help
ngc registry <subcommand> <action> --help
```

Add `--format_type json` when running commands (not for `--help`). If a reference does not list an option you need, use `--help` for that subcommand or action.
