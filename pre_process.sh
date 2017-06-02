#/usr/bin/env bash

### Run N experiments to compare original last sentences from ROC story corpus to artifical ones (correct and wrong)

set -e

function gen_pair_file {
	local file=$1
	local type=$2
	local read_command=$3
	
	# First, create two files (right and wrong samples)
	$read_command $file | awk -F "	" '{if ($8 == 1) {print $6} else {print $7}}'   | perl -e '$a = join(" ", <STDIN>); $a =~ s/\b/ /g; $a =~ s/ +/ /g; $a =~ s/^[   ]//g; $a =~ s/ $//;print $a'  | sed 's/^ //' > $work_dir/right_${type}.dat
	$read_command $file | awk -F "	" '{if ($8 == 2) {print $6} else {print $7}}'   | perl -e '$a = join(" ", <STDIN>); $a =~ s/\b/ /g; $a =~ s/ +/ /g; $a =~ s/^[   ]//g; $a =~ s/ $//;print $a'  | sed 's/^ //' > $work_dir/wrong_${type}.dat		
	
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

	perl gen_training_files.PL $dev_file $work_dir/dev ".9,.1,0" ".dat" 1

	for type in train dev; do
		gen_pair_file $work_dir/dev_${type}.dat $type "cat"
	done

	gen_pair_file $test_file "test" "tail -n +2"

	for t in train test dev; do
		gen_features.PL -i $work_dir/${t}_pairs.dat -f $work_dir/features_list.dat -o $work_dir/${t}_features.dat -len
	done


	nl=`wc -l $work_dir/features_list.dat | awk '{print $1}'`
}

work_dir=$PWD
if (( "$#" < 2 )); then
	echo "Usage: $0 <dev_file> <test_file> <work_dir=$work_dir>"
	exit -1
fi

dev_file=$1
test_file=$2

if (( "$#" > 2 )); then
	work_dir=$3
fi

if [ ! -d $work_dir ]; then
	mkdir $work_dir
fi

pre_process $work_dir $dev_file $test_file
