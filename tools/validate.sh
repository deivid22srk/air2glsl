#!/bin/sh
# Compiles every generated shader with glslangValidator (Vulkan 1.3) in
# parallel and reports how many are valid, plus the most common errors.
#
# usage: tools/validate.sh <out-dir> [max-errors-to-show]
set -eu

dir=${1:?usage: tools/validate.sh <out-dir> [n]}
show=${2:-12}
jobs=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# The dumper encodes the stage in the file name as
# shader_<section>_<index>_<stage>.glsl, so glslangValidator can be told the
# stage without a side channel.
cat >"$work/one.sh" <<'ONE'
file=$1
stage=$(basename "$file" .glsl | sed 's/.*_//')
case "$stage" in
    fragment) stage=frag ;;
    vertex) stage=vert ;;
    kernel) stage=comp ;;
    *) stage=frag ;;
esac
log="$file.log"
if glslangValidator -S "$stage" --target-env vulkan1.3 "$file" >"$log" 2>&1
then
    echo ok
else
    echo "ERR $(grep -o 'ERROR: .*' "$log" | head -1 | cut -c1-110)"
fi
rm -f "$log"
ONE

find "$dir" -name '*.glsl' >"$work/files"
xargs -n 1 -P "$jobs" sh "$work/one.sh" <"$work/files" >"$work/results" \
    2>/dev/null || true

ok=$(grep -c '^ok$' "$work/results" || true)
bad=$(grep -c '^ERR' "$work/results" || true)
printf 'compiled:  %s\nfailed:    %s\n' "$ok" "$bad"
if [ "$bad" -gt 0 ]; then
    printf 'top error messages:\n'
    grep '^ERR' "$work/results" | sed 's/^ERR //' | sort | uniq -c | \
        sort -rn | head -n "$show"
fi