# Q32 source reconstruction

The Q32 GPU result ran from `/gs/work/gemma-ibs-certified-eos-a0-20261128T100436Z`
at local base `59b16364f08f8d1a3d020a5a65da427b2cc656a3`. The four live
source files have since been generalized for fixed Q64/Q128; they are not the
sealed Q32 bytes. The Q32 deployment and n164 receipts are unchanged.

`multitier-reverse-base.sha256` identifies the four patch inputs.
`restore-q32.patch` goes **from those four multi-tier files to the sealed
Q32 files**. `q32-deployment.sha256` is the byte-for-byte manifest from the
Q32 deployment. Its top-level `controls/` entry is a deployed path, not a
Git path; stage the unchanged `registration.md` at that path in the temporary
reconstruction only. The pinned ordinary IBS runtime is not in the patch.

From the repository root, reconstruct into a fresh directory without
changing the checkout:

```bash
set -eu
root=$PWD
reg="$root/experiments/t2t/serving_slots/controls/gemma-factors/ibs-certified-eos"
shasum -a 256 -c "$reg/multitier-reverse-base.sha256"
stage=$(mktemp -d)
while read -r hash path; do
    mkdir -p "$stage/${path%/*}"
    cp "$root/$path" "$stage/$path"
done < "$reg/multitier-reverse-base.sha256"
deploy_reg=controls/gemma-factors/ibs-certified-eos
mkdir -p "$stage/$deploy_reg"
cp "$reg/registration.md" "$stage/$deploy_reg/"
(
    cd "$stage"
    git apply --check "$reg/restore-q32.patch"
    git apply "$reg/restore-q32.patch"
    shasum -a 256 -c "$reg/q32-deployment.sha256"
)
printf 'Verified Q32 sources: %s\n' "$stage"
```

The procedure was tested in a scratch directory: all five deployment
checksums matched after reconstruction.
