all: gflops.eps pct.eps sgemm.eps

gflops.eps: gflops.gplot times.dat
	gnuplot gflops.gplot

pct.eps: pct.gplot times.dat
	gnuplot pct.gplot

sgemm.eps: sgemm.gplot sgemm.dat
	gnuplot sgemm.gplot

times.dat: trans.dat magma_nn.dat tdot.dat
	echo "# n     trans1           trans2       magma_sgemm_nn cublas_sgemm_nn tdot" >$@
	paste $^ >>$@

sgemm.dat: magma_nn.dat magma_nt.dat sgemm_nn.dat sgemm_nt.dat test_nt.dat
	echo "# magma_nn cublas_nn magma_nt cublas_nt sgemm_nn sgemm_nt test_nt" >$@
	awk '{print $$1}' trans.dat | paste - $^ >>$@
