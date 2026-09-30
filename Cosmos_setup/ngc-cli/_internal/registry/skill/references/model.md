# Registry artifact: model

Versioning: Models have versions. Target format: org/[team/]model or org/[team/]model:version. Upload = upload-version; download = download-version. After upload-version, use commit-version to finalize.

Append --format_type json to commands that return output.

## Read operations

### list

List models or model versions. Pattern supports wildcards * and ?.

```bash
ngc registry model list [pattern] --format_type json
```

Options:
pattern — Optional. Format: org/[team/]name[:version]. Wildcards * and ? in name and version. For team-scoped listing, use a quoted namespace pattern such as `"org/team/*"`; do not rely on `--team` as an artifact ownership filter.
--column <column> — Output columns. Append for multiple
--sort <order> — Sort model versions (e.g. SEMVER_DESC)
--access-type <access_type> — Filter by access type
--product-name <product_name> — Filter by product name. Append for multiple
--policy — Policy filter
--signed — Show only models that are signed
--release-type — Release type filter.

Examples:

```bash
ngc registry model list "<org-name>/<team-name>/*" --format_type json
ngc registry model list "<org-name>/<team-name>/<model-name>:*" --format_type json
```

### info

Retrieve metadata for a model or model version.

```bash
ngc registry model info <target> [options] --format_type json
```

Options:
target — org/[team/]model_name[:version]
--files — List files in addition to details for a version
--credentials — List model credentials in addition to details for a version
--metrics — Deprecated; use --credentials.

### download-version (download)

Download a model version.

```bash
ngc registry model download-version <target> [options] --format_type json
```

Options:
target — org/[team/]model_name[:version]. Latest if version omitted
--dest <path> — Destination. Default: .
--file <wildcard> — Include only matching files. Append for multiple
--exclude <wildcard> — Exclude matching files/dirs. Append for multiple.

### download-version-signature

Download version signature.

```bash
ngc registry model download-version-signature <target> [options] --format_type json
```

Options:
target — org/[team/]model_name:version
--dest <path> — Destination. Default: .

### public-key

Download the public key used to sign models.

```bash
ngc registry model public-key <target> [options] --format_type json
```

Options:
target — org/[team/]model_name
--dest <path> — Destination. Default: .

### playground info

Model playground info, when available in the user's configured environment.

```bash
ngc registry model playground info <target> --format_type json
```

## Write operations

### create

Create a model. Requires application, framework, format, precision, short-desc.

If the model already exists, do not update it as a substitute for create. Report that the model exists and ask the user to explicitly request an update if they want metadata changed.

```bash
ngc registry model create <target> [options] --format_type json
```

Options:
target — org/[team/]model_name
--application <app> — Model application. Required
--framework <fwk> — Framework. Required
--format <fmt> — Format of the model. Required
--precision <prec> — Precision. Required
--short-desc <desc> — Short description. Required
--overview-filename <path> — Overview file
--bias-filename <path> — Bias file
--explainability-filename <path> — Explainability file
--privacy-filename <path> — Privacy file
--safety-security-filename <path> — Safety and security file
--display-name <name> — Display name
--label <label> — Label. Append for multiple
--label-set <label-set> — Label set. Format: org/[team/]name. Append for multiple
--logo <url> — Logo URL
--public-dataset-name, --public-dataset-link, --public-dataset-license — Public dataset info
--built-by <name>, --publisher <name> — Builder and publisher
--encryption-key-id <key-id> — Use existing encryption key (org/team scoped)
Mutex with --encryption-key-description
--encryption-key-description <description> — Create new encryption key for model.

### update

Update model or model version metadata.

```bash
ngc registry model update <target> [options] --format_type json
```

Options:
target — org/[team/]model_name[:version]
--application, --framework, --format, --precision, --short-desc, --desc — Model/version fields
--overview-filename, --bias-filename, --explainability-filename, --privacy-filename, --safety-security-filename — File paths
--display-name, --add-label, --remove-label, --label, --label-set, --logo — Metadata and labels
--public-dataset-name, --public-dataset-link, --public-dataset-license, --built-by, --publisher — Dataset and attribution
--gpu-model, --memory-footprint, --num-epochs, --batch-size, --accuracy-reached — Version-only fields
--set-latest — Set this version as latest
--policy, --release-type — Policy and release type
Mutex: --label/--label-set vs --add-label/--remove-label.

### upload-version (upload)

Upload a model version. Use commit-version to finalize, or --no-commit to allow more uploads.

```bash
ngc registry model upload-version <target> [options]
```

Options:
target — org/[team/]model_name:version
--source <path> — Source directory or file. Default: .
--gpu-model, --memory-footprint, --num-epochs, --batch-size, --accuracy-reached — Version metadata
--desc <desc> — Description
--dry-run — List paths and size without uploading
--link-type <type>, --link <url> — Link to resource/toolset (e.g. NGC, Github, Other)
--credentials-file <file> — JSON file with name and attributes. Append for up to 3 files
--metrics-file <file> — Deprecated; use --credentials-file
--base-version <version> — Base version to include files from; source overwrites overlapping
--no-commit — Do not complete version; allow subsequent uploads.

### commit-version

Finalize a model version after upload-version.

```bash
ngc registry model commit-version <target> --format_type json
```

Options:
target — org/[team/]model_name:version.

### remove

Remove a model or model version.

Removing a model target without `:version` removes the top-level model metadata. For requests such as "remove all versions" or "remove versions matching X", keep the model metadata: list versions first with a concrete model and version pattern, parse JSON, confirm the exact version targets, then remove each `org/[team/]model:version` sequentially.

```bash
ngc registry model remove <target> [options] --format_type json
```

Options:
target — org/[team/]model_name[:version]
-y, --yes — Run without interactive prompts (yes assumed).

### sign

Request a model version to be signed.

```bash
ngc registry model sign <target> --format_type json
```

Options:
target — org/[team/]model_name:version.

### update-license-terms

Update license terms for a model.

```bash
ngc registry model update-license-terms <target> [options] --format_type json
```

Options:
target — org/[team/]model_name[:version]
--license-terms-file <filename> — License terms file
--clear — Clear license terms
Mutex with --license-terms-file.

### accept-license

Display license terms for a model and accept all required terms after confirmation.

```bash
ngc registry model accept-license <target> [options]
```

Options:
target — org/[team/]model_name
-y, --yes — Automatically say yes to the license acceptance prompt.
--format_type json — Print structured license and acceptance output.

### deploy

Deploy subcommand: create, start, info, list, update, remove.

```bash
ngc registry model deploy list --format_type json
ngc registry model deploy start <target> [options] --format_type json
```

Options for deploy start/create: --gpu, --gpu-type, --disk, --dry-run, etc. Use ngc registry model deploy <action> --help for details.
