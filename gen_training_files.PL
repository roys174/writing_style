#!/usr/bin/env perl

### Generate train/dev/test files of a given size.

use strict;
use warnings;
use IO::File;
use List::Util qw/shuffle/;

sub main(@) {
	my $sizes = "40000,5000,5000";
	my $if = shift;
	my $suf = ".dat";
	my $skip = undef;
	my $of_pref = shift or die "Usage: $0 <ifs> <of_pref> <sizes=$sizes> <suf=$suf> <skip first line (optional)\n";
	
	if (@_) {
		$sizes = shift;
		if (@_) {
			$suf = shift;
			if (@_) {
				$skip = shift;
			}
		}
	}
	
	my @sizes = split(",", $sizes);
	
	my @lines;
	
	my $ifh = new IO::File($if) or die "Can't open $if for reading";
	
	# Skip first line?
	if (defined $skip) {
		$ifh->getline();
	}

	while (my $line = $ifh->getline()) {
		push(@lines, $line);
	}
		
	@lines = shuffle(@lines);
	
	my $n = scalar(@lines);

	if ($sizes[0] < 1) {
		@sizes = map {$sizes[$_]*$n } 0..2;
	}
		
	my @suffices = ("train", "dev", "test");
	
	foreach my $i (0 .. 2) {
		if ($sizes[$i] > 0) {
			my $of = $of_pref."_".$suffices[$i].$suf;
			my $ofh = new IO::File($of, "w") or die "Can't open $of for writing";
			foreach my $j (0 .. $sizes[$i]-1) {
				my $l = shift(@lines);
				$ofh->print($l);
			}
			$ofh->close();
		}		
	}
	
	return 0;
}

exit(main(@ARGV));
