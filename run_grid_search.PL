#!/usr/bin/env perl

use warnings;
use strict;
use IO::File;

### Run grid search for a given setup

sub main(@) {
	my $work_dir = shift or die "Usage: $0 <work_dir>\n";
	
	# Compute number of features for model.
	my $ifh = new IO::File("$work_dir/features_list.dat") or die "Can't open features file $work_dir/features_list.dat for reading";	
	my @e = $ifh->getlines();
	my $n_features = @e;
	$ifh->close();
	
	my $argmax = undef;
	my $max = 0;
	my $max2 = 0;
	
	# Traverse learning rate values.
	foreach my $c (qw/0.05 0.025 0.01 0.0075 0.00675 0.005 0.004 0.003 0.0025 0.001/) {
		# Test $c on dev set
		my ($exp1_result, $roc_result) = test_c($work_dir, $n_features, $c, 0);

		printf("Testing C=%f. On dev set: Experiment 1 results=%.03f, ROC story results=%.03f\n", $c, $exp1_result, $roc_result);
	
		# Tuning C on experiment 1 result. In case of tie, using ROC story results.
		if ($exp1_result > $max) {
			$max = $exp1_result;
			$max2 = $roc_result;
			$argmax = $c;
		} elsif ($exp1_result == $max and $roc_result >= $max2) {
			$max2 = $roc_result;
			$argmax = $c;
		}
	}
	
	print "Selected C is $argmax\n";
	
	test_c($work_dir, $n_features, $argmax,1);
	run_test($work_dir, $argmax, "test", 1);
	
	return 0;
}

sub run_test($$$$) {
	my $work_dir = shift;
	my $c = shift;
	my $name = shift;
	my $verbose = shift;

	my $output = `python2 test.py ${work_dir}/${name}_features.dat ${work_dir}/model.bin`;
	
	my @elements = split("\n", $output);

	my $roc_result = pop(@elements);

	my $exp1_result = pop(@elements);
	
	if ($verbose) {
		print "\nResults on $name set:\n";
		printf ("Experiment 1 scores: %s\n",$exp1_result);
		printf("ROC story classification scores: %s\n", $roc_result);
	}

	return ($roc_result, $exp1_result);
}

sub test_c($$$$) {
	my $work_dir = shift;
	my $n_features = shift;
	my $c = shift;
	my $verbose = shift;

	my $t = `python3 train.py -i ${work_dir}/train_features.dat -n $n_features -o  ${work_dir}/model.bin -c $c`;
	
	die if ($?);
		
	run_test($work_dir, $c, "train", $verbose);

	my ($exp1_result, $roc_result) = run_test($work_dir, $c, "dev", $verbose);
	
	chomp($roc_result);
	my @exp1_result_elements = split(" ", $exp1_result);
	my @roc_reuslts_elements = split(" ", $roc_result);

	return (pop(@roc_reuslts_elements), pop(@exp1_result_elements));
}


exit(main(@ARGV));
