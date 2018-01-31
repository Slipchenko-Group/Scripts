for f in $(ls ./efp/efp_files); do
	filename=$(echo $f | cut -d'.' -f 1);
	mkdir $filename-ba;
	cd $filename-ba;
	mkdir ref
	mkdir sat
	cd ref
	mkdir output
	cd ../sat
	mkdir output
	cd ..
	mkdir sat_results
	mkdir rmsd_results
	cp ../tools/create_mc_inp.py ./;
	cp ../tools/mc_run_filler ./;
	cp ../10_efp/$filename.efp.xyz ./;
	cp ../10_efp/efp_files/$f ./;
	#mv $f f1ref.efp
	cp ../tools/ba.efp sat/
	cp ../tools/ba.efp ref/
	cp $f ./ref/f1ref.efp
	cp ../tools/ba.efp ./ref;
	cp ../tools/ba.efp .sat;
	cp ../10_efp/${filename}_saturated.efp ./;
	cp ${filename}_saturated.efp ./sat/f1sat.efp
	#cp ../tools/create_inp_footer.py ./;
	#cp ../tools/parsing.py ./;
	cp ../tools/create_efp_inp.py ./;
	#cp ../tools/header ./;
	cp ../tools/create_bash_job.py ./;
	cp ../tools/extract_results.py ./;
	cp ../tools/calculate_rmsd.py ./;
	cp ../tools/min_rmsd.py ./;
	python create_mc_inp.py $f;
	#python create_inp_footer.py $filename
	/group/lslipche/apps/gamess/gamess_2016R1/rungms mc_run.inp 700 1 > mc_run.log;
	#python parsing.py;
	python create_efp_inp.py;
	python create_bash_job.py;
	cd ref;
	bash bash.sh; 
	cd ../sat;
	bash bash.sh;
	cd ..;
	python extract_results.py;
	python min_rmsd.py $filename.efp.xyz
	rm -rf sat;
	rm -rf ref;
	rm ~/scr/mc_run*;
	cd ..

	#cd ..
	#rm -rf $filename-ba
done