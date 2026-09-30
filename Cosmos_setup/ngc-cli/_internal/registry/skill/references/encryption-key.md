# Registry artifact: encryption-key

Encryption keys are used to lock models (and optionally other artifacts) at creation or when empty. Scoped to org/team; guest mode not allowed.

Append --format_type json to commands that return output.

## Required preflight

Before any encryption-key command, run `ngc config current`. If no account is configured, do not run `ngc registry encryption-key ...`; tell the user to run `ngc config set` in their own terminal and retry after configuration.

Do not use `--org` with encryption-key commands; they use the configured org. Do not retrieve or offer to retrieve API keys or credentials from NGC Vault, secret stores, files, or other systems to fill missing config.

## Read operations

### list

List encryption keys in the configured org/team scope.

```bash
ngc registry encryption-key list [options] --format_type json
```

Options:
--list-team <team_name> — Team to list keys from (uses configured org). Omit for org-level keys only.

### info

Get encryption key details and associated artifacts.

```bash
ngc registry encryption-key info <encryption_key_target> [options] --format_type json
```

Options:
encryption_key_target — [org/[team/]]keyid
--artifact-type <artifact_type> — Filter by type. Choices: model. Append for multiple. If omitted, queries all supported artifact types (currently model).

### status

Get status of encryption key workflow operations.

```bash
ngc registry encryption-key status <status_url> [status_url ...] --format_type json
```

Options:
status_url — One or more status URLs returned from remove or disassociate.

## Write operations

### remove

Remove encryption key and all associated artifacts (async). Returns statusUrl for polling.

```bash
ngc registry encryption-key remove <encryption_key_target> [encryption_key_target ...] [options] --format_type json
```

Options:
encryption_key_target — One or more [org/[team/]]keyid
--no-wait — Return status URL immediately without waiting for completion
-y, --yes — Run without interactive prompts (yes assumed).

### disassociate

Disassociate models from encryption keys (async). Returns statusUrl for polling.

```bash
ngc registry encryption-key disassociate [options] --format_type json
```

Options:
--model <model> — Model to disassociate. Format: org/[team/]model_name. Required. Append for multiple
--no-wait — Return immediately with status URL
-y, --yes — Run without interactive prompts (yes assumed).
