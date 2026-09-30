# Registry artifact: resource

Versioning: Resources have versions. Target format: org/[team/]resource or org/[team/]resource:version. Upload = upload-version; download = download-version. Same pattern as model; commit to finalize when applicable.

Append --format_type json to commands that return output.

## Read operations

### list

List resources or resource versions. Pattern supports wildcards.

```bash
ngc registry resource list [pattern] --format_type json
```

Options:
pattern — Optional. [org/[team/]]name[:version]. Wildcards * and ?; version supports character expressions
--column <column> — Output columns. Append for multiple
--access-type <access_type> — Filter by access type
--product-name <product_name> — Filter by product name. Append for multiple
--policy, --signed, --release-type — Filters.

### info

Retrieve metadata for a resource or resource version.

```bash
ngc registry resource info <target> [options] --format_type json
```

Options:
target — org/[team/]resource_name[:version]
--files — List files for a version (version required).

### download-version (download)

Download a resource version.

```bash
ngc registry resource download-version <target> [options] --format_type json
```

Options:
target — org/[team/]resource_name[:version]
--dest <path> — Destination. Default: .
--file <wildcard> — Include only matching files. Append for multiple
--exclude <wildcard> — Exclude matching files/dirs. Append for multiple.

### download-version-signature

Download the signature file of a resource version.

```bash
ngc registry resource download-version-signature <target> [options] --format_type json
```

Options:
target — org/[team/]resource_name:version
--dest <path> — Destination. Default: .

### public-key

Download the public key used to sign resources.

```bash
ngc registry resource public-key [options] --format_type json
```

Options:
--dest <path> — Destination. Default: .

## Write operations

### create

Create resource metadata. Requires application, framework, format, precision, short-desc.

If the resource already exists, do not update it as a substitute for create. Report that the resource exists and ask the user to explicitly request an update if they want metadata changed.

```bash
ngc registry resource create <target> [options] --format_type json
```

Options:
target — org/[team/]resource_name
--application <app> — Application. Required
--framework <fwk> — Framework. Required
--format <fmt> — Format of generated model/checkpoint. Required
--precision <prec> — Precision. Required
--short-desc <desc> — Short description. Required
--overview-filename <path> — Overview file
--display-name, --label, --label-set, --logo — Metadata
--public-dataset-name, --public-dataset-link, --public-dataset-license — Public dataset
--built-by, --publisher — Attribution
--advanced-filename, --performance-filename, --quick-start-guide-filename, --setup-filename — Guide files.

### update

Update resource or resource version metadata.

```bash
ngc registry resource update <target> [options] --format_type json
```

Options:
target — org/[team/]resource_name[:version]
Same as create plus version-only: --desc, --accuracy-reached, --batch-size, --gpu-model, --memory-footprint, --num-epochs, --release-notes-filename
--add-label, --remove-label, --label, --label-set
Mutex: --label/--label-set vs --add-label/--remove-label
--policy, --release-type — Policy and release type.

### upload-version (upload)

Upload a resource version.

```bash
ngc registry resource upload-version <target> [options]
```

Options:
target — org/[team/]resource_name:version
--source <path> — Source directory or file. Default: .
--accuracy-reached, --batch-size, --gpu-model, --memory-footprint, --num-epochs — Version metadata
--desc — Description
--performance-filename, --quick-start-guide-filename, --setup-filename, --release-notes-filename — Guide files
--dry-run — List paths and size without uploading
--base-version <version> — Base version to include files from; source overwrites.

### remove

Remove a resource or resource version.

Removing a resource target without `:version` removes the top-level resource metadata. For requests such as "remove all versions" or "remove versions matching X", keep the resource metadata: list versions first with a concrete resource and version pattern, parse JSON, confirm the exact version targets, then remove each `org/[team/]resource:version` sequentially.

```bash
ngc registry resource remove <target> [options] --format_type json
```

Options:
target — org/[team/]resource_name[:version]
-y, --yes — Run without interactive prompts (yes assumed).

### sign

Request a resource version to be signed.

```bash
ngc registry resource sign <target> --format_type json
```

Options:
target — org/[team/]resource_name:version.

### update-license-terms

Update license terms for a resource.

```bash
ngc registry resource update-license-terms <target> [options] --format_type json
```

Options:
target — org/[team/]resource_name[:version]
--license-terms-file <filename> — License terms file
--clear — Clear license terms
Mutex with --license-terms-file.

### accept-license

Display license terms for a resource and accept all required terms after confirmation.

```bash
ngc registry resource accept-license <target> [options]
```

Options:
target — org/[team/]resource_name
-y, --yes — Automatically say yes to the license acceptance prompt.
--format_type json — Print structured license and acceptance output.

### deploy

Deploy subcommand: create, start, info, list, update, remove.

```bash
ngc registry resource deploy <action> [options] --format_type json
```

Options for deploy: target, csp, --cpu, --gpu, --gpu-type, --disk, --ram, --image. Use ngc registry resource deploy <action> --help for details.
