# Registry artifact: image

Versioning: Image versions are called tags. Target format: org/[team/]image[:tag]. Upload = push; download = pull.

Append --format_type json to commands that return output.

## Read operations

### list

List container images accessible by the user. Pattern supports `*` and `?`. To list all tags for an image use `org/[team/]image:*`.

```bash
ngc registry image list [pattern] --format_type json
```

Options:
pattern — Optional. <org>/[<team>/]<image>[:<tags>]. Wildcards `*` and `?` supported
--column <column> — Output columns (e.g. created, description, name, org, tag, size). Append for multiple
--signed — Show only container images that are signed
--access-type <access_type> — Filter by access type
--product-name <product_name> — Filter by product name. Append for multiple
--policy — Policy filter
--release-type — Release type filter.

### info

Display information about an image repository or tagged image.

```bash
ngc registry image info <image> [options] --format_type json
```

Options:
image — <org>/[<team>/]<image>[:<tag>]
--layers — Show the layers of a tagged image
--history — Show the history of a tagged image
--details — Show the details of an image repository
--scan — Show the scan details of a tagged image. If no tag provided, most recent tag is used
Mutex: only one of --layers/--details or --history.

### pull

Pull a container image from the NGC image registry. If no tag is provided, latest is assumed.

For pull/auth troubleshooting, `ngc diag server --debug` validates registry reachability through the Docker Registry `/v2/` endpoint and logs the advertised bearer-token endpoint. Guest mode can confirm registry reachability, but private image pulls still require configured credentials and access.

```bash
ngc registry image pull <image> [options]
```

Options:
image — <org>/[<team>/]<image>[:<tag>]
--scan <file> — Download the image scan report as a CSV file instead of pulling the image.

### scan

Scan a container image from the NGC image registry. If no tag is provided, latest is used.

```bash
ngc registry image scan [pattern] --format_type json
```

Options:
pattern — Optional. <org>/[<team>/]<image>:<tag>.

### publickey

Return the public key used to sign images for local validation. No target.

```bash
ngc registry image publickey --format_type json
```

## Write operations

### remove

Remove an image repository or specific image with an image tag.

If the user asks to remove tags/versions while keeping the image repository metadata, include a tag in the target. Image removal supports tag globs for a concrete repository, so `org/team/image:*` removes matching tags without deleting the top-level repository metadata. If the user asks to remove the image repository itself, use the untagged image target after confirmation.

```bash
ngc registry image remove <pattern> [options] --format_type json
```

Options:
pattern — <org>/[<team>/]<image>[:<tags>]
-y, --yes — Automatically say yes to all interactive questions.

### update

Update image repository metadata.

```bash
ngc registry image update <image> [options] --format_type json
```

Options:
image — <org>/[<team>/]<image>[:<tag>]
--desc <desc> — Description for the target image
--overview <file.md> — Documentation (text or markdown file) for the image
--add-label <add-label> — Label to add. Append for multiple. Not with --label/--label-set
--remove-label <remove-label> — Label to remove. Append for multiple
--label <label> — Label to declare. Append for multiple. Not with --add-label/--remove-label
--label-set <label-set> — Label set to declare. Format: org/[team/]name. Append for multiple
--logo <url> — URL pointing to the logo for the repository
--publisher <publisher> — The person or entity publishing the image
--built-by <name> — The person who built the container image
--display-name <name> — Different name to display for the image
--multinode — Marks an image as supporting multinode
--no-multinode — Marks an image as not supporting multinode
--policy — Policy (image-specific)
--release-type — Release type (image-specific).

### set-latest

Set the specified tag as the latest tag of the repository.

```bash
ngc registry image set-latest <image> --format_type json
```

Options:
image — <org>/[<team>/]<image>:<tag>. Target image tag to set as latest (tag required).

### push

Push a container image to the NGC image registry. If no tag is provided, latest is assumed.

```bash
ngc registry image push <image> [options]
```

Options:
image — <org>/[<team>/]<image>[:<tag>]
-y, --yes — Automatically say yes to tagging the image if not already tagged in correct format
--desc <desc> — Description
--overview <file.md> — Documentation file
--label <label> — Label. Append for multiple
--label-set <label-set> — Label set. Format: org/[team/]name. Append for multiple
--logo <url> — Logo URL
--publisher <publisher> — Publisher
--built-by <name> — Built by
--multinode — Marks an image as supporting multinode
--display-name <name> — Display name.

### create

Create a top level metadata repository in the NGC image registry. Tags are added via push.

```bash
ngc registry image create <image> [options] --format_type json
```

Options:
image — <org>/[<team>/]<image>. Name of the image repository
--desc <desc> — Description
--overview <file.md> — Documentation file
--label <label> — Label. Append for multiple
--label-set <label-set> — Label set. Format: org/[team/]name. Append for multiple
--logo <url> — Logo URL
--publisher <publisher> — Publisher
--built-by <name> — Built by
--multinode — Marks an image as supporting multinode
--display-name <name> — Display name.

### sign

Have the image cryptographically signed by NVIDIA.

```bash
ngc registry image sign <image> --format_type json
```

Options:
image — <org>/[<team>/]<image>[:<tags>].

### update-license-terms

Update license terms for an image.

```bash
ngc registry image update-license-terms <target> [options] --format_type json
```

Options:
target — <org>/[<team>/]image_name[:tag]
--license-terms-file <filename> — License terms file
--clear — Clear license terms
Mutex with --license-terms-file.

### accept-license

Display license terms for an image and accept all required terms after confirmation.

```bash
ngc registry image accept-license <target> [options]
```

Options:
target — <org>/[<team>/]image_name
-y, --yes — Automatically say yes to the license acceptance prompt.
--format_type json — Print structured license and acceptance output.

### deploy

Manage interactive container deployments for images. Subcommand: ngc registry image deploy <action> [options].
Deploy actions:
start — Create interactive deployment of an image to a CSP. Args: target, csp; options: --gpu, --gpu-type, --disk, --dry-run
create — Create default deployment parameters
remove — Remove deployment
info — Deployment info
update — Update deployment parameters
list — List deployments
Example: ngc registry image deploy list --format_type json.
