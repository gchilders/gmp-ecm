#!/bin/bash
# gpu_compare.sh NEW_ECM REF_ECM
#
# Runs GPU stage 1 with fixed sigmas on inputs of many sizes and checks
# that the saved residues match a reference: REF_ECM (e.g. a CGBN build)
# where it accepts the size, otherwise gpu_ref_ladder.py on every curve.
NEW=${1:?new ecm}; REF=${2:?reference ecm}
B1=${B1:-2000}
# GPUWRAP: prefix for GPU runs only (e.g. a script that pauses a keepalive)
GPUWRAP=${GPUWRAP:-}
# REFCURVES: how many curves the (slow) Python reference checks
REFCURVES=${REFCURVES:-8}
CURVES=${CURVES:-64}
T=$(mktemp -d)
fail=0

# expressions with assorted sizes, including 2^k +- 1 (long runs of 1s/0s)
inputs=(
  "2^61-1" "(2^127-1)*(2^89-1)" "2^223-1" "2^251-1" "2^256+1"
  "(10^90+1)*7" "2^349-1" "2^512+1" "2^521-1" "2^607-1"
  "2^1018-1" "(2^1536-1)/(2^512-1)" "2^1279-1" "3^1000+2"
  "2^2203-1" "7^1000-6" "2^3217-1" "2^4253-1" "2^6000+1"
  "2^9689-1" "2^12200+1" "2^12283-1" "2^16000+3" "3^10000+2" "2^24570+1"
)
inputs+=( ${EXTRA_INPUTS} )
[ -n "$INPUTS" ] && inputs=( ${INPUTS} )

sig=1000
for e in "${inputs[@]}"; do
  sig=$((sig + 997))
  echo "$e" | $GPUWRAP $NEW -gpu -gpucurves $CURVES -sigma 3:$sig -save $T/new $B1 0 > $T/new.log 2>&1
  rc_new=$?
  echo "$e" | $GPUWRAP $REF -gpu -gpucurves $CURVES -sigma 3:$sig -save $T/ref $B1 0 > $T/ref.log 2>&1
  rc_ref=$?
  if [ -s $T/ref ] && [ $rc_ref -ne 1 ]; then
    grep -o 'X=0x[0-9a-f]*' $T/new > $T/xn; grep -o 'X=0x[0-9a-f]*' $T/ref > $T/xr
    if cmp -s $T/xn $T/xr && [ $rc_new = $rc_ref ] && [ -s $T/xn ]; then
      echo "OK   $e (vs ref, $(wc -l < $T/xn) curves, rc $rc_new)"
    else
      echo "FAIL $e (vs ref) rc new=$rc_new ref=$rc_ref"; fail=1
      tail -3 $T/new.log
    fi
  else
    # reference can't do it: check every curve against the independent
    # Python ladder (gpu_ref_ladder.py, next to this script)
    if python3 "$(dirname "$0")/gpu_ref_ladder.py" --all "$e" $sig $B1 $T/new $T/new.log $REFCURVES > $T/py.log; then
      echo "OK   $e (vs python reference, rc $rc_new)"
    else
      echo "FAIL $e (vs python reference)"; fail=1; tail -3 $T/py.log
    fi
  fi
  rm -f $T/new $T/ref
done
rm -rf $T
[ $fail = 0 ] && echo "ALL MATCH" || echo "MISMATCHES"
exit $fail
