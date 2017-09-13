#!/usr/bin/env perl -w

### Add language model's features to style features.

use strict;
use IO::File;

sub main(@) {
	my $lm_feautres_file = shift;
	my $features_if = shift;
	my $mapping_file = shift;
	my $n_feats = shift;
	my $of = shift or die "Usage: $0 <Language model features file> <Features input file> <question/id mapping file> <total number of style features> <Features output file>\n";
	
	my $mapping = read_mapping_file($mapping_file);

	my $mapping2 = read_lm_file($lm_feautres_file, $mapping);

	add_lm_features($mapping2, $features_if, $of, $n_feats);
	
	return 0;
}

# adding language model features to a given feature values file.
sub add_lm_features($$$$) {
	my $mapping = shift;
	my $if = shift;
	my $of = shift;
	my $n_feats = shift;
	
	my $f1 = $n_feats+1;
	my $f2 = $f1+1;
	my $f3 = $f2+1;
	
	print "Reading $if\n";
	my $ifh = new IO::File($if) or die "Can't open $if for reading";
	my $ofh = new IO::File(">$of") or die "Can't open $of for writing";
	
	while (my $line = $ifh->getline()) {
		chomp($line);
		
		my @e = split("\t", $line);
		
		my $text = $e[1];
		
		# Handle non-english characters
		$text =~ s/_[A-Z\$\.\',`:]++//g;
		$text =~ s/ ([^a-zA-Z0-9\$]++)/$1/g;
		$text =~ s/([^a-zA-Z0-9 ,]++) /$1/g;
		$text =~ tr/é//d;
		$text =~ tr/ //d;
		$text =~ s/[^\w]+$//;
		
		unless (exists $mapping->{$text}) {
			warn "Can't find '$text ($e[1])' in mapping\n";
			next;
		}
		
		$e[2] .= sprintf(" %d:%.03g %d:%.03g %d:%.03g\n",${f1},$mapping->{$text}->[0], ${f2},$mapping->{$text}->[1], ${f3},$mapping->{$text}->[2]);
		
		$ofh->print(join("\t", @e));
	}
	
	$ofh->close();
	$ifh->close();
} 

# Read file with language model scores. Assuming lines with sentence id, followed by three scores: P(s), P(s|c), P(s|c) - p(s).
# Return sent -> lm_scores mapping.
sub read_lm_file($$$$) {
	my $if = shift;
	my $mapping = shift;
	
	my %mapping;
	my $ifh = new IO::File($if) or die "Can't open $if for reading";
	
	# Skip first line.
	$ifh->getline();
	
	while (my $line = $ifh->getline()) {
		chomp($line);
		
		my ($id, $ps, $psc, $label) = split(",", $line);
		
		unless (exists $mapping->{$id}) {
			die "Cannot find id $id";
		} elsif (@{$mapping->{$id}} != 2) {
			die "$id != 2";
		}
		my $sent = $mapping->{$id}->[$label-1];

		die "cannot find $id,$label" unless defined $sent;
		
		$sent =~ tr/ //d;
		
		$sent =~ s/\W+$//;
		$sent =~ tr/é//d;
				
		$mapping{$sent} = [$ps,$psc,$psc-$ps];
	}
	
	return \%mapping;
}

# Read ROC story dataset, which maps ids to questions. Return id -> sentences mapping 
sub read_mapping_file($) {
	my $if = shift;
	
	my %mapping;
	my $ifh = new IO::File($if) or die "Can't open $if for reading";
	
	# Skip first line.
	$ifh->getline();
	
	while (my $line = $ifh->getline()) {
		chomp($line);
		
		my @e = split("\t", $line);
		my $id = shift(@e);
		my $res = pop(@e);
		my @sents = splice(@e, -2);
		
		if ($res == 2) {
			@sents = reverse(@sents);
		}
		
		$mapping{$id} = \@sents;
}
	
	return \%mapping;
}

exit(main(@ARGV));