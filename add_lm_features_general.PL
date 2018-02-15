#!/usr/bin/env perl

### Add language model's features to style features.
### This is a general version that adds any number of LM feaures. It expects each line to contain the sentence id, followed by a sequence of feature values, and then the label. 

use warnings;
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
	
	my $f1 = $n_feats;
	
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
		
		my @elements = ();
		
		foreach my $i (1..@{$mapping->{$text}}) {
			push(@elements, sprintf("%d:%.03g", $f1+$i, $mapping->{$text}->[$i-1]));
		}
		my $str = join(" ", @elements);
		
		$e[2] .= " ".$str."\n";
		
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
		
		my @fields = split(",", $line);
		
		my $id = shift(@fields);
		my $label = pop(@fields);
		
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
				
		$mapping{$sent} = \@fields;
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
