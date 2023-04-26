**PSSM Promoter Tool**

The tool applies CORPSE (Codon Restrained Promoter Silencing) method and inverted CORPSE (iCORPSE) to the provided gene sequence.

-35 and -10 promoters along with the additional non-canonical sequence motifs are predicted based on the Salis Lab Promoter Calculator (https://github.com/hsalis/SalisLabCode/tree/master/Promoter_Calculator).
Position-specific scoring matrix (PSSM) is applied to all the synonymous codon variants of the promoters associated with the lowest and highest transcription rates in order to maximally decrease (CORPSE) or increase the transcription rate (iCORPSE).
The output CSV file/files contain synonymous codon promoters and sequence motifs for the minimal and maximal transcriptional rates along with the non-canonical sequence motifs for forward and reverse strands.

INSTALLATION:

USAGE:

The tool requires a text or fasta file with a nucleotide sequence of a gene to process.
```
python3 PSSMPromoterCalculator.py <file_name>
```

Depending on the result, up to four output CSV files can be generated:
1) PSSMPromoterCalculator_MAX_FWD_results.csv - contains promoters to minimise transcription rate (forward strand)
2) PSSMPromoterCalculator_MAX_REV_results.csv - contains promoters to minimise transcription rate (reverse strand)
3) PSSMPromoterCalculator_MIN_FWD_results.csv - contains promoters to maximise transcription rate (forward strand)
4) PSSMPromoterCalculator_MIN_REV_results.csv - contains promoters to maximise transcription rate (reverse strand).

The output file fields:
new_sequence - contains a gene sequence (nt) with substituted promoters. Empty for the original promoters.
promoter_sequence
TSS -  transcriptional start site
Tx_rate - transcription initiation rate
UP - a 20-nucleotide region that appears upstream of the −35 motif, called the UP element
hex35 -  an upstream 6-nucleotide site called the −35 motif
PSSM_hex35 - position-specific scoring matrix value for the -35 motif
AA_hex35 - an amino acid sequence for the -35 motif
spacer - a spacer region that separates the −10 and −35 motifs
hex10 - a downstream 6-nucleotide site called the −10 motif
PSSM_hex10 - position-specific scoring matrix value for the -10 motif
AA_hex10 - an amino acid sequence for the -10 motif
disc - a typically 6-nucleotide region in between the −10 motif and TSS, called the discriminator (Disc)
ITR - the first 20 transcribed nucleotides, called the initial transcribed region (ITR)
dG_total
dG_10
dG_35
dG_disc
dG_ITR
dG_ext10
dG_spacer
dG_UP
dG_bind
UP_position
hex35_position
spacer_position
hex10_position
disc_position

References:

1. Logel DY, Trofimova E, Jaschke PR. Codon-Restrained Method for Both Eliminating and Creating Intragenic Bacterial Promoters. ACS Synth Biol. 2022 Jan 19;acssynbio.1c00359. Available from https://pubs.acs.org/doi/10.1021/acssynbio.1c00359. doi: 10.1021/acssynbio.1c00359
2. LaFleur TL, Hossain A, Salis HM. Automated model-predictive design of synthetic promoters to control transcriptional profiles in bacteria. Nat Commun. 2022 Sep 2;13(1):5159. Available from https://www.nature.com/articles/s41467-022-32829-5. doi: 10.1038/s41467-022-32829-5