# Registry artifact: csp (Cloud Service Provider)

CSP commands manage cloud service providers and their deployment constraints (GPU, disk, etc.) for registry deploy flows.

Append --format_type json to commands that return output.

## Read operations

### list

List CSPs for the configured org/team.

```bash
ngc registry csp list [options] --format_type json
```

Options:
--enabled-only — Show only enabled CSPs.

### info

Get CSP info.

```bash
ngc registry csp info <name> --format_type json
```

Options:
name — CSP name.

### info-settings

Get CSP deployment constraint settings.

```bash
ngc registry csp info-settings <name> --format_type json
```

Options:
name — CSP name.

## Write operations

### create

Create a new CSP entry.

```bash
ngc registry csp create <name> [options] --format_type json
```

Options:
name — New CSP key
--description <description> — Description
--display-name <display_name> — Display name
--logo <logo> — Logo URL
--enable — Make the CSP available for deployment
--label <label> — Label. Append for multiple.

### update

Update an existing CSP entry.

```bash
ngc registry csp update <name> [options] --format_type json
```

Options:
name — CSP entry to update
--description, --display-name, --logo, --label — Metadata. Append --label for multiple
--enable — Make CSP available for deployment
--disable — Make CSP unavailable for deployment.

### remove

Remove an existing CSP.

```bash
ngc registry csp remove <name> [options] --format_type json
```

Options:
name — CSP to remove
-y, --yes — Run without interactive prompts (yes assumed).

### create-settings

Create CSP deployment constraints (GPU and disk ranges, types, defaults).

```bash
ngc registry csp create-settings <name> [options] --format_type json
```

Options:
name — CSP to create constraints for
--gpu-min <gpu_min> — Minimum allowed number of GPUs. Required
--gpu-max <gpu_max> — Maximum allowed number of GPUs. Required
--gpu-default <gpu_default> — Default number of GPUs
--gpu-type <gpu_type> — Allowed GPU card type(s). Required. Append for multiple
--gpu-default-type <gpu_default_type> — Default GPU type
--disk-min <disk_min> — Minimum disk (GB). Required
--disk-max <disk_max> — Maximum disk (GB). Required
--disk-default <disk_default> — Default disk (GB).

### update-settings

Update CSP deployment constraints.

```bash
ngc registry csp update-settings <name> [options] --format_type json
```

Options:
name — CSP to update
--gpu-min, --gpu-max, --gpu-default, --gpu-type, --gpu-default-type — GPU constraints
--disk-min, --disk-max, --disk-default — Disk constraints. At least one value required.

### remove-settings

Remove deployment constraints for a CSP.

```bash
ngc registry csp remove-settings <name> [options] --format_type json
```

Options:
name — CSP to remove constraints for
-y, --yes — Run without interactive prompts (yes assumed).
