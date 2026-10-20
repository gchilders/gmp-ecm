#!/bin/bash
# gpu_bench.sh NEW_ECM REF_ECM  -- stage 1 throughput (curves/s), new vs ref.
# REF is swept over -gpucurves and its best is reported; NEW uses its
# default (autotuned) curve count, plus the same sweep for reference.
NEW=${1:?new ecm}; REF=${2:?reference ecm}
SIZES=${SIZES:-"250 500 700 1000 1500 2000 3000 4000 6000 8000 12000"}
SWEEP=${SWEEP:-"2048 4096 8192 16384 32768"}

tput () { # ecm expr B1 [curves]
  local c=""; [ -n "$4" ] && c="-gpucurves $4"
  echo "$2" | $1 -gpu $c -v -sigma 3:1000 $3 0 2>&1 |
    awk '/Throughput:/ {t=$2} /Computing [0-9]+ Step 1 took/ {n=$2}
         /-bit arithmetic \(TPI=/ {k=$4 $5; gsub(/[(),]/,"",k)} END {printf "%s %s %s", t, n, k}'
}

printf "%6s %8s | %-24s | %-24s | %-24s | %s\n" bits B1 "ref best (curves/s @n)" "new default (@n)" "new best (@n)" "speedup" kernel
for b in $SIZES; do
  e="2^$((b - 1))+12345"
  if   [ $b -le 1000 ]; then B1=20000
  elif [ $b -le 4000 ]; then B1=4000
  else B1=1000; fi
  rb=0; rn=-; nb=0; nn=-
  for c in $SWEEP; do
    read t n <<< "$(tput $REF "$e" $B1 $c)"
    [ -n "$t" ] && awk "BEGIN{exit !($t > $rb)}" && { rb=$t; rn=$n; }
    read t n _ <<< "$(tput $NEW "$e" $B1 $c)"
    [ -n "$t" ] && awk "BEGIN{exit !($t > $nb)}" && { nb=$t; nn=$n; }
  done
  read nd ndn kcfg <<< "$(tput $NEW "$e" $B1)"
  best=$(awk "BEGIN{print ($nd > $nb) ? $nd : $nb}")
  sp=$(awk "BEGIN{if ($rb > 0) printf \"%.2fx\", $best / $rb; else print \"ref n/a\"}")
  printf "%6s %8s | %12s @%-10s | %12s @%-10s | %12s @%-10s | %-7s %s\n" $b $B1 $rb $rn $nd $ndn $nb $nn $sp "$kcfg"
done
