# Registry artifact: label-set

Label sets hold labels used to tag artifacts. Target format: org/[team/]name. Uses configured org/team for list.

Append --format_type json to commands that return output.

## Read operations

### list

List label sets in the configured namespace. Optional resource-type filter.

```bash
ngc registry label-set list [options] --format_type json
```

Options:
--column <column> — Output columns. Append for multiple
--resource-type <resourceType> — Filter global label sets by resource type (e.g. MODEL, RESOURCE, IMAGE, CHART).

### info

Get information about a label set.

```bash
ngc registry label-set info <target> [options] --format_type json
```

Options:
target — org/[team/]name
--resource-type <resourceType> — Filter by resource type when resolving.

## Write operations

### create

Create a label set. Requires display-name.

```bash
ngc registry label-set create <target> [options] --format_type json
```

Options:
target — org/[team/]name
--display-name <dispName> — Name to display for the label set. Required
--label <label> — Label (use quotes with spaces). Append for multiple.

### update

Update a label set.

```bash
ngc registry label-set update <target> [options] --format_type json
```

Options:
target — org/[team/]name
--display-name <dispName> — Display name
--label — Deprecated; use --add-label
--add-label <label> — Label to add. Append for multiple
--remove-label <label> — Label to remove. Append for multiple.

### remove

Remove a label set.

```bash
ngc registry label-set remove <target> [options] --format_type json
```

Options:
target — org/[team/]name
-y, --yes — Run without interactive prompts (yes assumed).
