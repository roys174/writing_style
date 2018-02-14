#/usr/bin/env bash

### Run N experiments to compare original last sentences from ROC story corpus to artifical ones (correct and wrong)

set -e

function gen_pair_file {
	local file=$1
	local type=$2
	local read_command=$3

	right_of=$work_dir/right_${type}.dat 
	wrong_of=$work_dir/wrong_${type}.dat 
	
	# First, create two files (right and wrong samples)
	$read_command $file | awk -F "	" '{if ($8 == 1) {print $6} else {print $7}}'   | perl -e '$a = join(" ", <STDIN>); $a =~ s/\b/ /g; $a =~ s/ +/ /g; $a =~ s/^[   ]//g; $a =~ s/ $//;print $a'  | sed 's/^ //' > $right_of
	$read_command $file | awk -F "	" '{if ($8 == 2) {print $6} else {print $7}}'   | perl -e '$a = join(" ", <STDIN>); $a =~ s/\b/ /g; $a =~ s/ +/ /g; $a =~ s/^[   ]//g; $a =~ s/ $//;print $a'  | sed 's/^ //' > $wrong_of
	
	# PoS tag both files.
	for j in right wrong; do
		python spacy_pos_tag.py $work_dir/${j}_${type}{,_tagged}.dat 
	done
	
	# Merge both files into 1 file with the following format:
	# 1 right sentence
	# 2 wrong sentence
	paste $work_dir/{right,wrong}_${type}_tagged.dat | awk -F "	" '{print "1	"$1"\n2	"$2}' > $work_dir/${type}_pairs.dat
}

function pre_process {
	local wdir=$1
	local dev_file=$2
	local test_file=$3
	local lm_dev=$4
	local lm_test=$5

	if [ ${dev_file: -4} == ".csv" ]; then
		new_file=`echo $dev_file | sed 's/.csv/.tsv/'`
		
		python csv2tsv.py $dev_file $new_file
		dev_file=$new_file
	fi

	perl gen_training_files.PL $dev_file $work_dir/dev ".9,.1,0" ".dat" 1

	for type in train dev; do
		gen_pair_file $work_dir/dev_${type}.dat $type "cat"
	done

	if [ ${test_file: -4} == ".csv" ]; then
		new_file=`echo $test_file | sed 's/.csv/.tsv/'`
		
		python csv2tsv.py $test_file $new_file
		test_file=$new_file
	fi

	gen_pair_file $test_file "test" "tail -n +2"

	for t in train test dev; do
		gen_features.PL -i $work_dir/${t}_pairs.dat -f $work_dir/features_list.dat -o $work_dir/${t}_features.dat -len
	done
	
	nl=`wc -l $work_dir/features_list.dat | awk '{print $1-2}'`
	
	if [ -n "$lm_test" ]; then
		for type in train dev; do
			mv $work_dir/${type}_features{,_orig}.dat
			add_lm_features.PL $lm_dev $work_dir/${type}_features_orig.dat $dev_file $nl $work_dir/${type}_features.dat
		done
		type=test
		mv $work_dir/${type}_features{,_orig}.dat
		add_lm_features.PL $lm_test $work_dir/${type}_features_orig.dat $test_file $nl $work_dir/${type}_features.dat

		for i in 1 2 3; do
			let "nl++"
			echo "##LM${i}##	$nl	-1	1" >> $work_dir/features_list.dat
		done
	fi
}

work_dir=$PWD
if (( "$#" < 2 )); then
	echo "Usage: $0 <dev_file> <test_file> <work_dir=$work_dir> <language model scores (dev set)> <language model scores (test set)>"
	echo ""
	echo "-- dev and test files are ROC story corpus development and test sets"
	echo ""
	echo "-- work_dir is where temporary files are located during running" 
	echo ""
	echo "-- Last two arguments are optional: the language model scores of the development and test set (both need to be provided to consider them in the computation)."
	echo "See README file for more details"
	exit -1
fi

dev_file=$1
test_file=$2

if (( "$#" > 2 )); then
	work_dir=$3
	if (( "$#" > 4 )); then
		lm_dev=$4
		lm_test=$5
	fi
fi

if [ ! -d $work_dir ]; then
	mkdir $work_dir
fi

pre_process $work_dir $dev_file $test_file $lm_dev $lm_test
