#!/bin/bash
# WSL, after ops.sh and official.sh: the number of differing cases per op (the case name up to its type)
for log in ops.log ops_linux.log; do
  echo "== $log"
  grep "^DIFF" "$HOME/$log" | awk '{print $2}' \
    | sed -E 's/_(float32|float16|int8|uint8|int16|int32|int64|f32|complex64|complex128|[0-9]).*//' | sed 's/:$//' \
    | sort | uniq -c | sort -rn
done
