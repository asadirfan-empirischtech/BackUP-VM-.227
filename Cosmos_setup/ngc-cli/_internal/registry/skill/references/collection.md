# Registry artifact: collection

Collections group images, models, resources, and charts. No version in target; format: org/[team/]collection_name.

Append --format_type json to commands that return output.

## Read operations

### list

List collections. Pattern supports wildcards * and ?.

```bash
ngc registry collection list [pattern] --format_type json
```

Options:
pattern — Optional. org/[team/]collection_name. For team-scoped listing, use a quoted namespace pattern such as `"org/team/*"`; do not rely on `--team` as an artifact ownership filter.
--column <column> — Output columns. Append for multiple
--access-type <access_type> — Filter by access type
--product-name <product_name> — Filter by product name. Append for multiple
--policy, --release-type — Filters.

Example:

```bash
ngc registry collection list "<org-name>/<team-name>/*" --format_type json
```

### info

Display information about a collection.

```bash
ngc registry collection info <target> --format_type json
```

Options:
target — org/[team/]collection_name.

### find

Get collections containing a specified artifact.

```bash
ngc registry collection find <artifact_type> <artifact_target> --format_type json
```

Options:
artifact_type — MODEL, CHART, RESOURCE, or IMAGE
artifact_target — org/[team/]artifact_name.

## Write operations

### create

Create a collection. Requires short-desc and category.

```bash
ngc registry collection create <target> [options] --format_type json
```

Options:
target — org/[team/]name
--short-desc <desc> — Brief description. Required
--category <category> — Use case. Required. Choices: from CollectionCategoryTypeEnum (e.g. APPLICATION, DATASET)
--display-name <name> — Human-readable name
--label, --label-set — Labels. Append for multiple
--logo <logo> — Logo URL
--overview-filename <path> — Markdown overview file
--built-by <name>, --publisher <publisher> — Owner and publisher
--add-image <image> — Image to include. Format: org/[team/]name. Append for multiple
--add-model <model> — Model to include. Append for multiple
--add-resource <resource> — Resource to include. Append for multiple
--add-chart <chart> — Chart to include. Append for multiple.

### update

Update a collection.

```bash
ngc registry collection update <target> [options] --format_type json
```

Options:
target — org/[team/]collection_name
Same as create (all optional): --display-name, --short-desc, --category, --logo, --overview-filename, --built-by, --publisher
--add-label, --remove-label, --label, --label-set — Labels. Mutex: --label/--label-set vs --add-label/--remove-label
--add-image, --add-model, --add-resource, --add-chart — Artifacts to add. Append for multiple
--remove-image, --remove-model, --remove-resource, --remove-chart — Artifacts to remove. Append for multiple.

### remove

Remove a collection.

```bash
ngc registry collection remove <target> [options] --format_type json
```

Options:
target — org/[team/]collection_name
-y, --yes — Run without interactive prompts (yes assumed).
