**PSSM Promoter Tool**

The tool applies CORPSE (Codon Restrained Promoter Silencing) method and inverted CORPSE (iCORPSE) to the provided gene sequence.

-35 and -10 promoters along with the additional non-canonical sequence motifs are predicted based on the Salis Lab Promoter Calculator (https://github.com/hsalis/SalisLabCode/tree/master/Promoter_Calculator).
Position-specific scoring matrix (PSSM) is applied to all the synonymous codon variants of the promoters associated with the lowest and highest transcription rates in order to maximally decrease (CORPSE) or increase the transcription rate (iCORPSE).
The output CSV file/files contain synonymous codon promoters and sequence motifs for the minimal and maximal transcriptional rates along with the non-canonical sequence motifs for forward and reverse strands.

For a web (Google Colab) version, please navigate to https://colab.research.google.com/drive/171iBNCrA1hMS-LpX_qaFani34HiueTZO?usp=sharing.

INSTALLATION:

1. Install required libraries using pip:
```
pip install pssm-promoter-tool
```

2. Download and unpack the archive with files using "Code"->"Download ZIP" buttons in the right corner at https://github.com/ellinium/pssm_promoter_tool. 
Or use
```
git clone https://github.com/ellinium/pssm-promoter-tool
```



USAGE:

The tool requires a text or fasta file with a nucleotide sequence of a gene to process.
From the folder with the downloaded files run:
```
python pssm_promoter_calculator.py <file_name>
```
where 'file_name' is a path to the file with a gene sequence (TXT or FASTA format).

Depending on the result, up to four output CSV files can be generated:
1) PSSMPromoterCalculator_MAX_FWD_results.csv - contains promoters to minimise transcription rate (forward strand)
2) PSSMPromoterCalculator_MAX_REV_results.csv - contains promoters to minimise transcription rate (reverse strand)
3) PSSMPromoterCalculator_MIN_FWD_results.csv - contains promoters to maximise transcription rate (forward strand)
4) PSSMPromoterCalculator_MIN_REV_results.csv - contains promoters to maximise transcription rate (reverse strand).

The output file fields in the CSV files contain data from Salis' Promoter calculator and additional fields:
1) new_sequence - contains a gene sequence (nt) with substituted promoters. Empty for the original promoters.
2) promoter_sequence - contains -35 motif, spacer and - 10 motif
3) TSS -  transcriptional start site
4) Tx_rate - transcription initiation rate
5) UP - a 20-nucleotide region that appears upstream of the −35 motif, called the UP element
6) hex35 -  an upstream 6-nucleotide site called the −35 motif
7) PSSM_hex35 - position-specific scoring matrix value for the -35 motif
8) AA_hex35 - an amino acid sequence for the -35 motif
9) spacer - a spacer region that separates the −10 and −35 motifs
10) hex10 - a downstream 6-nucleotide site called the −10 motif
11) PSSM_hex10 - position-specific scoring matrix value for the -10 motif
12) AA_hex10 - an amino acid sequence for the -10 motif
13) disc - a typically 6-nucleotide region in between the −10 motif and TSS, called the discriminator (Disc)
14) ITR - the first 20 transcribed nucleotides, called the initial transcribed region (ITR)
15) dG_total - total Gibbs free energy for the sequence
16) dG_10 - -10 motif Gibbs free energy
17) dG_35 - -35 motif Gibbs free energy
18) dG_disc - a discriminator Gibbs free energy
19) dG_ITR - an ITR Gibbs free energy
20) dG_ext10 −10 extended motif Gibbs free energy
21) dG_spacer - a spacer Gibbs free energy
22) dG_UP - an UP Gibbs free energy
23) dG_bind - binding Gibbs free energy
24) UP_position - a position of the UP element
25) hex35_position - a position of the -35 motif
26) spacer_position - a position of the spacer
27) hex10_position - a position of the -10 motif
28) disc_position - a position of the discriminator

References:

1. Logel DY, Trofimova E, Jaschke PR. Codon-Restrained Method for Both Eliminating and Creating Intragenic Bacterial Promoters. ACS Synth Biol. 2022 Jan 19;acssynbio.1c00359. Available from https://pubs.acs.org/doi/10.1021/acssynbio.1c00359. doi: 10.1021/acssynbio.1c00359
2. LaFleur TL, Hossain A, Salis HM. Automated model-predictive design of synthetic promoters to control transcriptional profiles in bacteria. Nat Commun. 2022 Sep 2;13(1):5159. Available from https://www.nature.com/articles/s41467-022-32829-5. doi: 10.1038/s41467-022-32829-5