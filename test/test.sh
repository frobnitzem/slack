#!/bin/sh
cd /Users/rogers/work/traj_an/src/parray/test
TESTS="test_ast test_gc test_parse test_dot test_eval test_dir test_serial"
# end of test.sh (running the tests)

fail=""
for t in $TESTS; do
    echo " ==================== $t ===================="
    ./$t || {
        echo "$t returned $?"
        fail="$fail $t"
    }
    echo
done
echo " ==================== Summary: ===================="

if [[ x"" == x"$fail" ]]; then
    echo "All tests passed!"
else
    echo "Failed tests: $fail"
    exit 0
fi

