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
[! you need to have a Git account and be authorised in the system to run the command !]


USAGE:

The tool requires a text or fasta file with a nucleotide sequence of a gene to process.
From the folder with the downloaded files run:
```
python pssm_promoter_calculator.py <file_name>
```
where 'file_name' is a path to the file with a gene sequence (TXT or FASTA format).

By default, the command processes 20 PSSM promoter combinations with the highest PSSM scores for transcription rate maximisation or 20 combinations with the lowest PSSM scores for transcription rate minimisation.
To run all the promoter combinations, use an additional argument 'all':
```
python pssm_promoter_calculator.py <file_name> all
```


Depending on the result, up to four output CSV files can be generated:
1) PSSMPromoterCalculator_MIN_FWD_results.csv - contains promoters to minimise transcription rate (forward strand)
2) PSSMPromoterCalculator_MIN_REV_results.csv - contains promoters to minimise transcription rate (reverse strand)
3) PSSMPromoterCalculator_MAX_FWD_results.csv - contains promoters to maximise transcription rate (forward strand)
4) PSSMPromoterCalculator_MAX_REV_results.csv - contains promoters to maximise transcription rate (reverse strand).

The output file fields in the CSV files contain data from Salis' Promoter calculator and additional fields:
1) Type - 'Original Promoter' - for original sequence promoters; 'Modified Promoter' - for synonymous promoters. 
2) TSS -  transcriptional start Site position (nt) 
3) Tx_rate - transcription unitiation rate (au)
4) Tx_rate_FoldChange - the fold change between the original transcription rate and re-calculated with PSSM primers
5) frame - a reading frame for the promoters 
6) hex35 - -35 hexamer sequence (an upstream 6-nucleotide site called the −35 motif)
7) PSSM_hex35 - position-specific scoring matrix value for the -35 motif 
8) hex35_sequence - contains 9nt sequence that includes -hex35 promoter if it's in the 2nd or 3rd frame
9) hex35_aa - an amino acid sequence for the -35 motif 
10) hex10 - -10 hexamer sequence (a downstream 6-nucleotide site called the −10 motif)
11) PSSM_hex10 - position-specific scoring matrix value for the -10 motif 
12) hex10_sequence - contains 9nt sequence that includes -hex35 promoter if it's in the 2nd or 3rd frame
13) hex10_aa - an amino acid sequence for the -10 motif 
14) UP - an upstream sequence (UP). A 20-nucleotide region that appears upstream of the −35 motif.
15) spacer - a spacer sequence that separates the −10 and −35 motifs 
16) disc - discriminator sequence. A typically 6-nucleotide region in between the −10 motif and TSS. 
17) ITR - the first 20 transcribed nucleotides called the initial transcribed region (ITR)
18) new_gene_sequence - contains a gene sequence (nt) with substituted (Type = 'Modified Promoter') or  original promoters (Type = 'Original Promoter')
19) promoter_sequence - contains -35 motif, spacer and - 10 motif 
20) dG_total - total Gibbs free energy change (kcal/mol)
21) dG_10 - Gibbs binding free Energy for -10 hexamer (kcal/mol)
22) dG_35 - Gibbs binding free Energy for -35 hexamer (kcal/mol)
23) dG_disc - Gibbs free energy penalty for non-optimal discriminator element (kcal/mol)
24) dG_ITR - Gibbs free energy change for R-loop formation at the initial transcribed region (kcal/mol)
25) dG_ext10 - Gibbs binding free Energy for the extended -10 hexamer (kcal/mol)
26) dG_spacer - Gibbs free energy penalty for non-optimal spacing (kcal/mol)
27) dG_UP - Gibbs binding free energy for the upstream element (kcal/mol)
28) dG_bind - binding Gibbs free energy 
29) UP_position - a position of the UP element 
30) hex35_position - a position of the -35 motif 
31) spacer_position - a position of the spacer 
32) hex10_position - a position of the -10 motif 
33) disc_position - a position of the discriminator

References:

1. Logel DY, Trofimova E, Jaschke PR. Codon-Restrained Method for Both Eliminating and Creating Intragenic Bacterial Promoters. ACS Synth Biol. 2022 Jan 19;acssynbio.1c00359. Available from https://pubs.acs.org/doi/10.1021/acssynbio.1c00359. doi: 10.1021/acssynbio.1c00359
2. LaFleur TL, Hossain A, Salis HM. Automated model-predictive design of synthetic promoters to control transcriptional profiles in bacteria. Nat Commun. 2022 Sep 2;13(1):5159. Available from https://www.nature.com/articles/s41467-022-32829-5. doi: 10.1038/s41467-022-32829-5