# Registry artifact: chart

Versioning: Charts have versions. Target format: org/[team/]chart_name[:version]. Upload = push; download = pull.

Append --format_type json to commands that return output.

## Read operations

### list

List charts or chart versions. Pattern supports wildcards * and ? and version character expressions.

```bash
ngc registry chart list [pattern] --format_type json
```

Options:
pattern — Optional. org/[team/]chart_name[:version]
--column <column> — Output columns. Append for multiple
--access-type <access_type> — Filter by access type
--product-name <product_name> — Filter by product name. Append for multiple
--policy, --release-type — Filters.

### info

Retrieve metadata for a chart or chart version.

```bash
ngc registry chart info <target> [options] --format_type json
```

Options:
target — org/[team/]chart_name[:version]
--files — List files for a version (version required).

### pull (download)

Download a chart version.

```bash
ngc registry chart pull <target> [options]
```

Options:
target — org/[team/]chart[:version]. Latest if version omitted
--dest <path> — Destination. Default: .

## Write operations

### create

Create chart metadata. Versions are added via push.

If the chart already exists, do not update it as a substitute for create. Report that the chart exists and ask the user to explicitly request an update if they want metadata changed.

```bash
ngc registry chart create <target> [options] --format_type json
```

Options:
target — org/[team/]chart_name
--short-desc <shortDesc> — Brief description. Required
--overview-filename <path> — Overview file
--display-name, --label, --label-set, --logo, --built-by, --publisher — Metadata.

### update

Update chart or chart version metadata.

```bash
ngc registry chart update <target> [options] --format_type json
```

Options:
target — org/[team/]chart_name[:version]
--overview-filename, --display-name, --short-desc — Description fields
--add-label, --remove-label, --label, --label-set — Labels. Mutex: --label/--label-set vs --add-label/--remove-label
--built-by, --publisher, --logo — Attribution and logo
--policy, --release-type — Policy and release type.

### push (upload)

Push (upload) a chart version.

```bash
ngc registry chart push <target> [options]
```

Options:
target — org/[team/]chart_name:version. Standard format; .tgz filename format deprecated
--source <path> — Directory or packaged chart path. Default: .
--dry-run — List paths and size without uploading.

### remove

Remove a chart or chart version.

Removing a chart target without `:version` removes the top-level chart metadata and all chart versions. For requests such as "remove all versions" or "remove versions matching X", keep the chart metadata: list versions first with a concrete chart and version pattern, parse JSON, confirm the exact version targets, then remove each `org/[team/]chart:version` sequentially.

```bash
ngc registry chart remove <target> [options] --format_type json
```

Options:
target — org/[team/]chart_name[:version]. If no version, removes all versions then chart metadata
-y, --yes — Run without interactive prompts (yes assumed).

### update-license-terms

Update license terms for a chart.

```bash
ngc registry chart update-license-terms <target> [options] --format_type json
```

Options:
target — org/[team/]chart_name[:version]
--license-terms-file <filename> — License terms file
--clear — Clear license terms
Mutex with --license-terms-file.

### accept-license

Display license terms for a chart and accept all required terms after confirmation.

```bash
ngc registry chart accept-license <target> [options]
```

Options:
target — org/[team/]chart_name
-y, --yes — Automatically say yes to the license acceptance prompt.
--format_type json — Print structured license and acceptance output.
