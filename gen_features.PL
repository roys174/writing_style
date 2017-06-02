#!/usr/bin/env perl

### Generate features for logistic regression model

use strict;
use warnings;
use Getopt::Long;
use IO::File;
use Term::ANSIColor;

my %CW_TAGS = ("N" => undef, "V" => undef, "J" => undef, "R" => undef);

sub main(@) {
	my $min_thr = 5;
	my $max_patt_len = 5;
	my $if;
	my $feature_file;
	my $of;
	my $use_len;
	my $char_ngrams=4;
	my $dont_normalize;
	
	usage({
		"if=s" => \$if,
		"dont_normalize+"				=> \$dont_normalize,
		"feature_file=s"				=> \$feature_file,
		"of=s"							=> \$of,
		"max_patt_len=i" 				=> \$max_patt_len,
		"min_thr=i"						=> \$min_thr,
		"char_ngrams=i"					=> \$char_ngrams,
		"len+"							=> \$use_len,
	}, ["if=s","of=s","feature_file=s"]);
	
	$max_patt_len--;
	my %saved;

	my $max_len = 0;
	
	# File exists -- read features.
	if (-e $feature_file) {
		my $ifh = new IO::File($feature_file) or die "Can't open $feature_file for reading";
		
		$max_len = $ifh->getline();
		chomp($max_len);
		while (my $line = $ifh->getline()) {
			chomp($line);
			my ($patt, $n, $sum, $max) = split("	", $line);
			$saved{$patt} = [$n, $max];
		}
		
		$ifh->close();
	# File doesn't exist -- generate features. 
	} else {	
		my %patts;
		my $ifh = new IO::File($if) or die "Can't open $if for reading";

		my $is_first = 1;
		while (my $line = $ifh->getline()) {
			chomp($line);
	
			my ($label, $text) = split("\t", $line);
	
			my @w = split(" ", $text);
			
			if (defined $use_len and @w > $max_len) {
				$max_len = @w;
			}

			my $patts = get_patts(\@w, $max_patt_len, $char_ngrams,$dont_normalize);

			foreach my $patt (keys %$patts) {
				unless (exists $patts{$patt}) {
					$patts{$patt} = [$patts->{$patt}, $patts->{$patt}];
				} else {
					$patts{$patt}->[0] += $patts->{$patt};
					if ($patts{$patt}->[1] < $patts->{$patt}) {
						$patts{$patt}->[1] = $patts->{$patt}
					}
				}
			}
			$is_first = 0;
		}
		$ifh->close();
		
		my $i = 0;
		my @sk = sort {$patts{$b}->[0] <=> $patts{$a}->[0]} keys %patts;

		my $ofh = new IO::File($feature_file, "w") or die "Can't open $feature_file for writing";
		
		$ofh->print($max_len."\n");
		foreach my $k (@sk) {
			last if $patts{$k}->[0] < $min_thr;
			$ofh->print("$k\t$i\t$patts{$k}->[0]\t$patts{$k}->[1]\n");
			$saved{$k} = [$i++, $patts{$k}->[1]];
		}
		$ofh->close();
	}

	my $n_feats = scalar(keys %saved)+defined($use_len);
	# print $n_feats." found\n";
	
	unless ($n_feats) {
		return 1;
	}

	
	my $ifh = new IO::File($if) or die "Can't open $if for reading";
	my $ofh = new IO::File($of, "w") or die "Can't open $of for writing";
	
	while (my $line = $ifh->getline()) {
		chomp($line);
		
		my ($label, $text) = split("\t", $line);
		
		my @w = split(" ", $text);

		my $patts = get_patts(\@w, $max_patt_len, $char_ngrams);
		
		my %ts;
		
		foreach my $patt (keys %$patts) {
			if (exists $saved{$patt}) {
				$ts{$saved{$patt}->[0]} = $patts->{$patt};
				unless (defined $dont_normalize) {
					$ts{$saved{$patt}->[0]} /= ($saved{$patt}->[1]);
				}	
			}
		}
		
		if ($use_len) {
			$ts{$n_feats-1} = scalar(@w);
			
			unless (defined $dont_normalize) {
				$ts{$n_feats-1} /= $max_len;
			}
		}
		
		$ofh->print("$label\t$text\t");		
		
		$ofh->print(join(" ", map { $_.":".sprintf("%.3g",$ts{$_})} keys %ts));
		$ofh->print("\n");
	}
	
	$ofh->close();
	$ifh->close();
		
	return 0;
}

sub get_patts($$$$) {
	my $w = shift;
	unshift(@$w, "START_UNK");
	my $max_patt_len = shift;
	my $char_ngrams = shift;
	my $dont_normalize = shift;
	
	my %patts;
	
	my @w = map { exists($CW_TAGS{substr((split("_", $_))[1],0,1)}) ? (split("_", $_))[1] : (split("_", $_))[0] } @$w;
	
	foreach my $i (0 .. $max_patt_len) {
		foreach my $j (0 .. @w-$i-1) {
			my $p = join(" ", @w[$j..$j+$i]);
			$patts{$p}++;
		}
	}
	
	if ($char_ngrams) {
		my @w2 = map { (split("_",$_))[0]}  @$w;
		my $l = join(" ", @w2);
	
		foreach my $i (0 .. length($l)-$char_ngrams) {
			$patts{substr($l,$i,$char_ngrams)}++;
		}
	}
	
	return \%patts;
}


sub usage($$) {
        my $options = shift;
        my $mandatory = shift;
        my $help;

        foreach my $k (@$mandatory) {
                if (ref $k eq "ARRAY") {
                        foreach my $k2 (@$k) {
                                unless (exists $options->{$k2}) {
                                        die "Mandatory option $k2 is not optional\n";
                                }
                        }
                } elsif (not exists $options->{$k}) {
                        die "Mandatory option $k is not optional\n";
                }
        }


        $options->{"h+"} = \$help;

        my $result = GetOptions(%$options);
         
          
        if (not $result or $help) {
                getopt_gen_message($options, $mandatory);
        }
        
        foreach my $k (@$mandatory) {
                if (ref $k eq "ARRAY") {
                        my $found = 0;
                        foreach my $k2 (@$k) {
                                if (defined ${$options->{$k2}}) {
                                        $found = 1;
                                        last;
                                }
                        }

                        unless ($found) {
                                getopt_gen_message($options, $mandatory);
                        }
                } elsif (not defined ${$options->{$k}}) {
                        getopt_gen_message($options, $mandatory);
                }
        }
}

sub getopt_gen_message($$) {
        my $options = shift;
        my $mandatory = shift;
        
        print $0."\n";
        
        foreach my $k (sort {$a cmp $b} keys %$options) {
                print "-$k";
                if (defined ${$options->{$k}}) {
                        print " (default '";
                        print color 'blue';
                        print ${$options->{$k}};
                        print color 'reset';
                        print "')";
                }
                
                foreach my $field (@$mandatory) {
                        if ($field eq $k and not defined ${$options->{$k}}) {
                                print color 'bold red';
                                print " [MANDATORY]";
                                print color 'reset';
                        }
                }
                print "\n";
        }
        
        die "\n";
}



exit(main(@ARGV));